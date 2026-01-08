import json
import numpy as np
import sounddevice as sd
import whisper

from src.minilm import MiniLMRetriever
from src.router import IntentRouter

# ===== 마이크/ASR 설정 =====
SAMPLE_RATE = 16000
RECORD_SECONDS = 3.5
INPUT_DEVICE = 1           # 너의 마이크. (WASAPI로 성공했으면 9로 바꾸기)
WHISPER_MODEL = "base.en"  # 느리면 "tiny.en"

# ===== 의미 추론 설정 =====
INTENT_BANK_PATH = "data/intent_bank_en.json"
THRESHOLD = 0.55

# 안전: STOP/START는 키워드 우선 처리(권장)
STOP_KEYWORDS = ["stop", "pause", "hold on", "wait"]
START_KEYWORDS = ["start", "begin", "continue"]


def load_intent_bank(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_audio(seconds: float) -> np.ndarray:
    print(f"[REC] Speak now ({seconds:.1f}s)...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE,
    )
    sd.wait()
    return audio.squeeze()


def override_intent(text: str):
    t = text.lower()
    if any(k in t for k in STOP_KEYWORDS):
        return {"intent": "STOP", "score": 1.0, "text": text}
    if any(k in t for k in START_KEYWORDS):
        return {"intent": "START", "score": 1.0, "text": text}
    return None


def main():
    print("[ASR] Loading Whisper:", WHISPER_MODEL)
    asr = whisper.load_model(WHISPER_MODEL)

    print("[NLU] Loading MiniLM retriever + router...")
    intent_bank = load_intent_bank(INTENT_BANK_PATH)
    retriever = MiniLMRetriever(intent_bank)   # sentence-transformers 기반
    router = IntentRouter(retriever, threshold=THRESHOLD)

    print("\nEnter=record, q=quit\n")

    while True:
        cmd = input("Command > ").strip().lower()
        if cmd == "q":
            break

        # 1) 마이크 녹음
        audio = record_audio(RECORD_SECONDS)

        # 2) ASR (음성 -> 텍스트)
        print("[ASR] Transcribing...")
        result = asr.transcribe(audio, language="en", fp16=False)
        text = (result.get("text") or "").strip()
        print("[TEXT]", text if text else "(empty)")

        if not text:
            print()
            continue

        # 3) 안전 키워드 우선
        forced = override_intent(text)
        if forced:
            print("\n[OVERRIDE]")
            print(forced)
            print()
            continue

        # 4) MiniLM 의미 추론(의도 라우팅)
        chosen, candidates = router.route(text)

        print("\n[Top-3 candidates]")
        for c in candidates:
            print(f"- {c['intent']:10s} score={c['score']:.3f} ({c['text']})")

        print("\n[CHOSEN]")
        print(chosen)
        print()


if __name__ == "__main__":
    main()
