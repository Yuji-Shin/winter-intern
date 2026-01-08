import json
import torch

from src.minilm import MiniLMRetriever
from src.router import IntentRouter

def load_intent_bank(path="data/intent_bank.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("\n종료: q / quit / exit\n")

    intent_bank = load_intent_bank()
    retriever = MiniLMRetriever(intent_bank)
    router = IntentRouter(retriever, threshold=0.55)

    while True:
        utt = input("환자 발화 > ").strip()
        if utt.lower() in ("q", "quit", "exit"):
            break
        if not utt:
            continue

        chosen, candidates = router.route(utt)

        print("\n[Top-3 candidates]")
        for c in candidates:
            print(f"- {c['intent']:10s} score={c['score']:.3f} ({c['text']})")

        print("\n[CHOSEN]")
        print(chosen)
        print()

if __name__ == "__main__":
    main()
