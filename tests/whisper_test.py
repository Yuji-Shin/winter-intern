import whisper

AUDIO_PATH = r"C:\Users\caubi\Documents\Sound Recordings\sample.m4a"   # 네 파일명이 다르면 여기만 바꾸기 (예: "Recording.m4a")
MODEL_NAME = "base.en"      # 느리면 "tiny.en" 추천

def main():
    print("[LOAD] model:", MODEL_NAME)
    model = whisper.load_model(MODEL_NAME)

    print("[ASR] transcribing:", AUDIO_PATH)
    result = model.transcribe(AUDIO_PATH, language="en", fp16=False)

    print("\n[TEXT]")
    print(result["text"].strip())

if __name__ == "__main__":
    main()
