import json
import os
import random
import time
import winsound

class IntentResponder:
    """
    intent_responses_en.json을 읽어서
    - mode=tts  -> speak.say(...)
    - mode=beep -> winsound.Beep(...)
    - mode=silent -> 아무것도 안함
    """

    def __init__(self, config_path: str, speak, event_logger=None):
        self.config_path = config_path
        self.speak = speak
        self.event = event_logger

        self.cfg = {}
        self._load()

        # (추가 디듑/쿨다운이 필요하면 여기서도 가능)
        self.last_intent = ""
        self.last_time = 0.0
        self.min_gap_sec = 0.25

    def _load(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    def _pick_text(self, texts, fallback=""):
        if not texts:
            return fallback
        if isinstance(texts, str):
            return texts
        return random.choice(texts)

    def respond(self, intent: str, *, heard_text: str = "", force: bool = False, **kwargs):
        """
        kwargs: placeholder 채우기용(예: total_moved=1.23)
        """
        if not intent:
            return

        now = time.time()
        if (not force) and (intent == self.last_intent) and (now - self.last_time) < self.min_gap_sec:
            return

        self.last_intent = intent
        self.last_time = now

        block = self.cfg.get(intent)
        if not block:
            return

        mode = (block.get("mode") or "silent").lower()

        # placeholder dict
        fmt = {"heard_text": heard_text}
        fmt.update(kwargs)

        if mode == "silent":
            return

        if mode == "beep":
            pattern = block.get("beep") or []
            if self.event:
                self.event.log(f"[RESPOND] {intent} -> BEEP")
            for freq, dur in pattern:
                try:
                    winsound.Beep(int(freq), int(dur))
                except Exception:
                    pass
            return

        if mode == "tts":
            text = self._pick_text(block.get("texts"), fallback="")
            try:
                text = text.format(**fmt)
            except Exception:
                # placeholder mismatch 나도 그냥 원문 말하기
                pass

            if self.event:
                self.event.log(f"[RESPOND] {intent} -> TTS: {text}")

            # speak.say는 네 코드의 TTSGuard
            self.speak.say(text, tag=f"{intent} TTS", force=force)
            return
