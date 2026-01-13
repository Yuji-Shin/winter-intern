import os
import json
import numpy as np
import sounddevice as sd
import whisper

from src.minilm import MiniLMRetriever
from src.router import IntentRouter

from src.response_generator import ResponseGenerator, DialogState  # 영어 버전 권장
from src.tts_kokoro import KokoroTTS
from src.audio_out import play
from src.logger import JsonlLogger


# ======================
# Audio / ASR settings
# ======================
SAMPLE_RATE = 16000
RECORD_SECONDS = 3.5

INPUT_DEVICE = None   # ✅ None이면 "기본 입력 장치" 사용. (문제 생기면 숫자로 지정)
OUTPUT_DEVICE = None  # ✅ None이면 "기본 출력 장치" 사용. (소리 안 나면 숫자로 지정)

WHISPER_MODEL = "base.en"  # 느리면 "tiny.en"

# ======================
# NLU settings
# ======================
INTENT_BANK_PATH = "data/intent_bank_en.json"
THRESHOLD = 0.55

STOP_KEYWORDS = ["stop", "pause", "hold on", "wait"]
START_KEYWORDS = ["start", "begin", "continue"]

# intent label normalize (네 intent_bank에 맞춰 수정 가능)
INTENT_MAP = {
    "PAIN": "pain",
    "DISCOMFORT": "discomfort",
    "ANXIETY": "anxiety",
    "pain": "pain",
    "discomfort": "discomfort",
    "anxiety": "anxiety",
}


def load_intent_bank(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_devices():
    """Optional: fix default input/output devices."""
    if OUTPUT_DEVICE is not None or INPUT_DEVICE is not None:
        sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)


def print_devices():
    print("\n=== sounddevice devices ===")
    for i, d in enumerate(sd.query_devices()):
        name = d["name"]
        inch = d["max_input_channels"]
        outch = d["max_output_channels"]
        sr = d.get("default_samplerate", None)
        print(f"[{i}] in={inch} out={outch} sr={sr} | {name}")
    print("default (input, output):", sd.default.device)
    print("===========================\n")


def record_audio(seconds: float) -> np.ndarray:
    """
    Robust recording:
    - If device only supports stereo input, record 2ch and convert to mono.
    - Returns mono float32 array.
    """
    print(f"[REC] Speak now ({seconds:.1f}s)...")

    # Resolve input device index
    in_dev = INPUT_DEVICE
    if in_dev is None:
        in_dev = sd.default.device[0]  # default input index

    devinfo = sd.query_devices(in_dev, "input")
    max_ch = int(devinfo["max_input_channels"])
    if max_ch <= 0:
        raise RuntimeError(
            f"Selected INPUT_DEVICE={in_dev} is not an input device. max_input_channels={max_ch}"
        )

    # Use 1ch if possible, otherwise 2ch then downmix
    channels = 1 if max_ch >= 1 else max_ch
    if max_ch >= 2:
        channels = 2

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        device=in_dev,
    )
    sd.wait()

    # (N, ch) -> (N,)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32).squeeze()


def override_intent(text: str):
    t = text.lower()
    if any(k in t for k in STOP_KEYWORDS):
        return {"intent": "STOP", "score": 1.0, "text": text}
    if any(k in t for k in START_KEYWORDS):
        return {"intent": "START", "score": 1.0, "text": text}
    return None


def normalize_intent(intent: str) -> str:
    if not intent:
        return "other"
    return INTENT_MAP.get(intent, "other")


def main():
    print("RUNNING FILE:", os.path.abspath(__file__))

    # 장치 강제 지정이 필요하면 여기서 적용
    setup_devices()

    # 디바이스 목록 보고 싶으면 주석 해제
    # print_devices()

    print("[ASR] Loading Whisper:", WHISPER_MODEL)
    asr = whisper.load_model(WHISPER_MODEL)

    print("[NLU] Loading MiniLM retriever + router...")
    intent_bank = load_intent_bank(INTENT_BANK_PATH)
    retriever = MiniLMRetriever(intent_bank)
    router = IntentRouter(retriever, threshold=THRESHOLD)

    # (D) English response generator + state
    rg = ResponseGenerator()
    state = DialogState()

    # (E) Kokoro (네가 성공한 설정)
    tts = KokoroTTS(lang_code="a", voice="af_heart", sr=24000)

    # (F) logger
    logger = JsonlLogger("logs/ivs_whisper_kokoro.jsonl")

    print("\nEnter=record | d=devices | q=quit\n")

    while True:
        cmd = input("Command > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "d":
            print_devices()
            continue

        # 1) Record
        audio = record_audio(RECORD_SECONDS)

        # 2) ASR
        print("[ASR] Transcribing...")
        result = asr.transcribe(audio, language="en", fp16=False)
        text = (result.get("text") or "").strip()
        print("[TEXT]", text if text else "(empty)")

        if not text:
            print()
            continue

        # 3) STOP/START override
        forced = override_intent(text)
        if forced:
            print("\n[OVERRIDE]", forced)

            if forced["intent"] == "STOP":
                response_text = "Okay. I will pause. Tell me when you are ready to continue."
            else:
                response_text = "Okay. We can continue. Tell me if anything feels uncomfortable."

            audio_out, sr = tts.synthesize(response_text)
            play(audio_out, sr)

            logger.log_turn(
                user_text=text,
                intent=forced["intent"],
                score=forced["score"],
                response_text=response_text,
                state=state.__dict__,
            )

            print("BOT>", response_text)
            print()
            continue

        # 4) NLU route
        chosen, candidates = router.route(text)

        print("\n[Top-3 candidates]")
        for c in candidates:
            print(f"- {c['intent']:12s} score={c['score']:.3f} ({c['text']})")

        print("\n[CHOSEN]", chosen)

        raw_intent = chosen.get("intent") if isinstance(chosen, dict) else None
        score = chosen.get("score", 0.0) if isinstance(chosen, dict) else 0.0
        intent = normalize_intent(raw_intent)

        # (D) Generate English response text
        response_text, state = rg.generate(text, intent, state)
        print("\n[BOT TEXT]", response_text)

        # (E) TTS + (F) play/log
        audio_out, sr = tts.synthesize(response_text)
        play(audio_out, sr)

        logger.log_turn(
            user_text=text,
            intent=intent,
            score=score,
            response_text=response_text,
            state=state.__dict__,
        )
        print()

    print("bye!")


if __name__ == "__main__":
    main()
