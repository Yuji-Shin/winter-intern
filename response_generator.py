import random
import re
from dataclasses import dataclass
from typing import Optional


TEMPLATES = {
    "pain": [
        "I’m sorry you’re in pain. I’ll reduce the pressure a little.",
        "Thanks for telling me. I’ll adjust the pressure to make it more comfortable.",
        "That sounds painful. I’ll try to be as gentle and quick as possible.",
    ],
    "discomfort": [
        "I understand. Let me adjust the position or pressure to reduce discomfort.",
        "Thanks—I'll make it more comfortable. Please take slow breaths.",
        "If it feels uncomfortable, we can pause anytime. Just let me know.",
    ],
    "anxiety": [
        "It’s okay to feel nervous. We’re doing this safely, step by step.",
        "If you’d like, I can explain each step as we go. You’re doing great.",
        "No worries—if anything feels uncomfortable, we can adjust right away.",
    ],
    "other": [
        "Thanks for telling me. Let me know if anything feels uncomfortable.",
        "Okay—tell me right away if you need a pause or adjustment.",
    ],
}


def extract_0_10(text: str) -> Optional[int]:
    # Extract a number 0~10 (e.g., "7", "7/10", "pain 8", "10")
    m = re.search(r"\b(10|[0-9])\b", text)
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 10 else None


@dataclass
class DialogState:
    last_question: Optional[str] = None  # "ask_pain_score"
    pain_score: Optional[int] = None


class ResponseGenerator:
    def generate(self, user_text: str, intent: str, state: DialogState):
        # If we previously asked pain score, interpret numeric answer
        if state.last_question == "ask_pain_score":
            v = extract_0_10(user_text)
            if v is not None:
                state.pain_score = v
                state.last_question = None
                return (
                    f"Got it—your pain is {v} out of 10. "
                    f"I’ll reduce the pressure and slow down a bit.",
                    state,
                )

        key = intent if intent in TEMPLATES else "other"
        base = random.choice(TEMPLATES[key])

        # Ask pain score once if not available
        if intent == "pain" and state.pain_score is None:
            state.last_question = "ask_pain_score"
            base += " On a scale from 0 to 10, how strong is the pain right now?"

        return base, state
