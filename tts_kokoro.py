from __future__ import annotations

import numpy as np
from kokoro import KPipeline


class KokoroTTS:
    """
    Kokoro TTS wrapper
    - lang_code: "a"  (네가 테스트에서 성공한 값)
    - voice: "af_heart"
    - sr: 24000
    """

    def __init__(self, lang_code: str = "a", voice: str = "af_heart", sr: int = 24000):
        self.lang_code = lang_code
        self.voice = voice
        self.sr = sr
        self.pipeline = KPipeline(lang_code=lang_code)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Returns:
          audio (float32, mono), sr
        """
        if not text.strip():
            # 빈 텍스트면 무음 대신 짧은 공백 처리
            text = " "

        chunks = []
        generator = self.pipeline(text, voice=self.voice)

        for _, _, audio in generator:
            # audio is numpy float array
            if audio is not None and len(audio) > 0:
                chunks.append(audio)

        if not chunks:
            raise RuntimeError("Kokoro returned no audio. (chunks empty)")

        out = np.concatenate(chunks, axis=0).astype(np.float32)
        return out, self.sr
