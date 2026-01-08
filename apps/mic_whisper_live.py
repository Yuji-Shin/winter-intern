import numpy as np
import sounddevice as sd
import whisper

# ===== 설정 =====
SAMPLE_RATE = 16000
RECORD_SECONDS = 3.5
WHISPER_MODEL = "base.en"   # 느리면 tiny.en

# 마이크가 여러 개면 여기에 index 지정 가능 (예: 1)
INPUT_DEVICE = 1  # None이면 기본 장치 사용


def record_audio(seconds: float) -> np.ndarray:
    print(f"[REC] Speak now ({seconds:.1f}s)...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE
    )
    sd.wait()
    return audio.squeeze()


def main():
    print("[ASR] loading whisper:", WHISPER_MODEL)
    model = whisper.load_model(WHISPER_MODEL)

    print("\nEnter=record, q=quit\n")
    while True:
        cmd = input("Command > ").strip().lower()
        if cmd == "q":
            break

        audio = record_audio(RECORD_SECONDS)

        print("[ASR] transcribing...")
        result = model.transcribe(audio, language="en", fp16=False)
        text = (result.get("text") or "").strip()
        print("[TEXT]", text if text else "(empty)")
        print()

if __name__ == "__main__":
    main()
