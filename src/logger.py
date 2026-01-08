import json
import time
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str = "logs/dialog.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_turn(self, user_text: str, intent: str, score: float, response_text: str, state: dict):
        item = {
            "ts": time.time(),
            "user_text": user_text,
            "intent": intent,
            "intent_score": float(score) if score is not None else None,
            "response_text": response_text,
            "state": state,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
