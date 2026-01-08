import sounddevice as sd
import numpy as np


def play(audio: np.ndarray, sr: int):
    """
    audio: float32 mono array
    """
    if audio is None or len(audio) == 0:
        return
    sd.play(audio, sr)
    sd.wait()
