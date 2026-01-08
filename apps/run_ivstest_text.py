from src.tts_kokoro import KokoroTTS
from src.response_generator import ResponseGenerator, DialogState
from src.audio_out import play
from src.logger import JsonlLogger


def fake_router(text: str):
    """
    Temporary router for English input.
    """
    t = text.strip().lower()

    # pain
    if any(k in t for k in ["pain", "hurt", "hurts", "aching", "ache", "sore", "too much", "it hurts"]):
        return "pain", 0.90

    # discomfort
    if any(k in t for k in ["uncomfortable", "discomfort", "pressure", "too tight", "too hard", "tight", "push"]):
        return "discomfort", 0.80

    # anxiety
    if any(k in t for k in ["nervous", "anxious", "scared", "worried", "afraid", "panic"]):
        return "anxiety", 0.80

    return "other", 0.50


def main():
    # Your successful Kokoro settings
    tts = KokoroTTS(lang_code="a", voice="af_heart", sr=24000)

    rg = ResponseGenerator()
    state = DialogState()
    logger = JsonlLogger()

    print("=== IVS Text Test (English) ===")
    print("Exit: q / quit / exit")
    print("------------------------------")

    while True:
        user_text = input("USER> ").strip()
        if user_text.lower() in ["q", "quit", "exit"]:
            break

        intent, score = fake_router(user_text)
        response_text, state = rg.generate(user_text, intent, state)

        audio, sr = tts.synthesize(response_text)
        play(audio, sr)

        logger.log_turn(
            user_text=user_text,
            intent=intent,
            score=score,
            response_text=response_text,
            state=state.__dict__,
        )

        print(f"BOT> {response_text}  (intent={intent}, score={score:.2f})")


if __name__ == "__main__":
    main()
