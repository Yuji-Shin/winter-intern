# final_combined.py
# Realtime Whisper (utterance-level) + MiniLM intent routing
# + Async RTDE motion worker (keeps listening while moving)
# + Intent -> fixed responses (TTS / Beep / Silent) via JSON config
# + TTS feedback-loop prevention: ASR is muted while TTS is speaking
#
# ✅ Key behavior
# - 로봇은 계속 듣고 있다가(ASR) "환자가 말했을 때"만 intent에 따라 응답(TTS/Beep)
# - STOP: Beep만 (루프 방지)
# - PAIN: 문장으로 TTS (필수)
# - TTS 재생 중에는 마이크 입력을 버려서(TTS->mic) "emergency stop" 반복 루프 방지
# - 주기적 브리핑(5초마다 말하기) 기본 OFF

import os
import sys
import csv
import json
import re
import time
import queue
import threading
import wave
import subprocess
import random
from datetime import datetime
from time import perf_counter

import numpy as np
import sounddevice as sd
import whisper
import winsound

import rtde_control
import rtde_receive

from src.minilm import MiniLMRetriever
from src.router import IntentRouter


# ======================
# Audio / ASR settings
# ======================
SAMPLE_RATE = 16000
INPUT_DEVICE = 1
OUTPUT_DEVICE = 2  # Microsoft Sound Mapper - Output (MME)

sd.default.device = (INPUT_DEVICE, OUTPUT_DEVICE)
print("sounddevice default (in,out) =", sd.default.device)

WHISPER_MODEL = "base.en"

CHUNK_MS = 30

# 오탐 많으면 더 올리기: 0.018~0.03
ENERGY_THRESHOLD = 0.022

MIN_SPEECH_SEC = 0.40
END_SILENCE_SEC = 0.70
MAX_UTT_SEC = 4.5
DEBUG_ENERGY = False


# ======================
# NLU (MiniLM) settings
# ======================
INTENT_BANK_PATH = "data/intent_bank_en.json"
THRESHOLD = 0.55


# ======================
# Robot settings
# ======================
HOST = "127.0.0.1"  # URSim IP (VM이면 VM IP로 변경)
BASE_SPEED = 0.25
BASE_ACC = 0.5
DRY_RUN = False


# ======================
# Safety / behavior
# ======================
PAIN_RETRACT_Z_M = 0.03
POSE_TOL_M = 0.001

XYZ_LIMITS = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (0.0, 1.2),
}

# 키워드 오버라이드(초저지연)
STOP_KEYWORDS = [
    "stop", "pause", "hold on", "wait", "halt", "freeze",
    "emergency stop", "e-stop", "estop"
]
START_KEYWORDS = ["start", "begin", "continue", "go", "resume", "arm"]
PAIN_KEYWORDS = [
    "it hurts", "hurt", "pain", "ouch", "painful", "it's painful",
    "that hurts", "that's painful", "too much pressure", "uncomfortable"
]

DIR_WORDS = {
    "up":       ("z", +1),
    "down":     ("z", -1),
    "left":     ("y", +1),
    "right":    ("y", -1),
    "forward":  ("x", +1),
    "back":     ("x", -1),
    "backward": ("x", -1),
}
MOVE_VERBS = ["move", "go", "shift", "translate", "slide"]


# ======================
# Speech policy switches
# ======================
ENABLE_STARTUP_TTS = False
ENABLE_PERIODIC_BRIEFING = False      # ✅ 기본 OFF (말 많아지는 원인)
SPEAK_BLOCKED_NOT_ARMED = False
SPEAK_UNHANDLED_INTENT = False

# TTS chatter 방지 (연속 발화 제한)
TTS_COOLDOWN_SEC = 0.35
TTS_DEDUP_SEC = 3.0


# ======================
# Logging
# ======================
def _now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


class EventCSV:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.f = open(path, "w", newline="", encoding="utf-8-sig")
        self.w = csv.writer(self.f)
        self.w.writerow(["timestamp", "event_tag"])
        self.f.flush()

    def log(self, tag: str):
        ts = _now_str()
        print(f"[{ts}] {tag}")
        with self.lock:
            self.w.writerow([ts, tag])
            self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class MetricsCSV:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.fields = [
            "ts", "kind", "step", "utt_id", "duration_ms", "asr_ms",
            "text", "override_intent", "chosen_intent", "chosen_score", "top3", "note",
        ]
        self.w = csv.DictWriter(self.f, fieldnames=self.fields)
        self.w.writeheader()
        self.f.flush()

    def log(self, **row):
        base = {k: "" for k in self.fields}
        base.update(row)
        base["ts"] = base["ts"] or _now_iso()
        with self.lock:
            self.w.writerow(base)
            self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# ======================
# sounddevice playback helpers
# ======================
def _load_wav_np(path: str):
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        frames = wf.readframes(n)

    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), sr

    if sw == 2:
        data = np.frombuffer(frames, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(frames, dtype=np.int32).reshape(-1, ch).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw}")

    if data.ndim == 1:
        data = data[:, None]

    # force stereo
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]

    return data, sr


def _resample_linear(stereo: np.ndarray, src_sr: int, dst_sr: int):
    if src_sr == dst_sr or stereo.shape[0] == 0:
        return stereo
    n = stereo.shape[0]
    new_n = max(1, int(n * dst_sr / src_sr))
    t_old = np.linspace(0.0, 1.0, n, endpoint=False)
    t_new = np.linspace(0.0, 1.0, new_n, endpoint=False)
    y0 = np.interp(t_new, t_old, stereo[:, 0])
    y1 = np.interp(t_new, t_old, stereo[:, 1])
    return np.column_stack([y0, y1]).astype(np.float32)


def play_stereo_force_device(stereo: np.ndarray, sr: int, device: int = OUTPUT_DEVICE):
    if stereo is None or stereo.size == 0:
        return
    out_sr = int(sd.query_devices(device, "output")["default_samplerate"])
    x = stereo.astype(np.float32)
    x = _resample_linear(x, sr, out_sr)
    sd.play(x, out_sr, device=device)
    sd.wait()


# ======================
# TTS (subprocess pyttsx3 -> WAV -> sounddevice)
# ======================
_TTS_PY = r"""
import sys
import pyttsx3
text = sys.argv[1]
wav_path = sys.argv[2]
engine = pyttsx3.init()
engine.save_to_file(text, wav_path)
engine.runAndWait()
"""

def render_tts_wav_subprocess(text: str, wav_path: str, timeout_sec: float = 8.0):
    cmd = [sys.executable, "-c", _TTS_PY, text, wav_path]
    try:
        subprocess.run(
            cmd, check=True, timeout=timeout_sec,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        return False
    return os.path.exists(wav_path) and os.path.getsize(wav_path) >= 1024


class SoundDeviceTTS:
    """
    - announce(text): 비동기 큐에 넣고,
    - 워커가 WAV 생성 -> OUTPUT_DEVICE로 재생
    """
    def __init__(self, event_logger: EventCSV, out_device: int = OUTPUT_DEVICE):
        self.event = event_logger
        self.out_device = out_device
        self.q = queue.Queue()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self.cache_dir = os.path.join("logs", "tts_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    @property
    def is_speaking(self):
        return self._speaking.is_set()

    def announce(self, text: str):
        if text:
            self.q.put(text)

    def wait_until_finished(self, timeout_sec: float = 20.0):
        t0 = time.time()
        while True:
            if self.q.empty() and (not self.is_speaking):
                return True
            if (time.time() - t0) > timeout_sec:
                return False
            time.sleep(0.05)

    def stop(self):
        self._stop.set()
        try:
            self.q.put_nowait("")
        except Exception:
            pass

    def _worker(self):
        idx = 0
        while not self._stop.is_set():
            try:
                msg = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._stop.is_set() or not msg:
                continue

            idx += 1
            self._speaking.set()

            wav_path = os.path.join(
                self.cache_dir, f"tts_{datetime.now():%Y%m%d_%H%M%S_%f}_{idx}.wav"
            )
            try:
                ok = render_tts_wav_subprocess(msg, wav_path, timeout_sec=8.0)
                if not ok:
                    self.event.log("[TTS] render FAILED (pyttsx3 missing?)")
                    continue
                audio, sr = _load_wav_np(wav_path)
                play_stereo_force_device(audio, sr, device=self.out_device)
            except Exception as e:
                self.event.log(f"[TTS] exception: {repr(e)}")
            finally:
                self._speaking.clear()
                try:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                except Exception:
                    pass


class BriefingWrapper:
    def __init__(self, event_logger: EventCSV):
        self.event = event_logger
        self.tts = SoundDeviceTTS(event_logger, out_device=OUTPUT_DEVICE)

    @property
    def is_speaking(self):
        return self.tts.is_speaking

    def announce(self, msg: str, tag: str = None):
        if tag:
            self.event.log(tag)
        if msg:
            self.tts.announce(msg)

    def wait_until_finished(self):
        ok = self.tts.wait_until_finished(timeout_sec=20.0)
        if not ok:
            self.event.log("[TTS] wait timeout (continuing)")

    def stop(self):
        self.tts.stop()


class TTSGuard:
    """
    - 짧은 시간에 계속 말하는 것 방지
    - 동일 문장 반복 방지
    """
    def __init__(self, briefing: BriefingWrapper):
        self.briefing = briefing
        self.last_text = ""
        self.last_time = 0.0

    def say(self, text: str, tag: str = None, force: bool = False):
        if not text:
            return
        now = time.time()

        if not force:
            if (now - self.last_time) < TTS_COOLDOWN_SEC:
                return
            if text == self.last_text and (now - self.last_time) < TTS_DEDUP_SEC:
                return

        self.last_text = text
        self.last_time = now
        self.briefing.announce(text, tag=tag)


# ======================
# Intent responses (JSON)
# ======================
RESPONSES_PATH = os.path.join("data", "intent_responses_en.json")

DEFAULT_INTENT_RESPONSES = {
    "_meta": {
        "lang": "en",
        "note": "intent -> response policy. mode: tts | beep | silent. placeholders: {heard_text}"
    },
    "STOP": {
        "mode": "beep",
        "beep": [[880, 150], [880, 150]],
        "texts": ["Emergency stop."]
    },
    "START": {
        "mode": "tts",
        "texts": ["Armed. Ready.", "Okay. I am ready to proceed."]
    },
    "MOVE_SLOW": {
        "mode": "tts",
        "texts": ["Okay. I will slow down.", "Understood. Moving slower."]
    },
    "EXPLAIN": {
        "mode": "tts",
        "texts": [
            "I am moving the tool based on your commands.",
            "I heard: {heard_text}. I will follow your instructions."
        ]
    },
    "PAIN": {
        "mode": "tts",
        "texts": [
            "I heard that you are in pain. I will stop and retract now.",
            "Okay. Stopping and retracting for safety."
        ]
    },
    "OK": {
        "mode": "tts",
        "texts": ["Okay. Thank you. I will continue.", "Understood. Continuing."]
    },
    "RULE_MOVE": {
        "mode": "silent",
        "texts": ["Moving now."]
    },
    "SEQUENCE": {
        "mode": "tts",
        "texts": ["Starting sequence.", "Starting the routine now."]
    }
}

def ensure_intent_responses_file(event: EventCSV):
    if os.path.exists(RESPONSES_PATH):
        return
    os.makedirs(os.path.dirname(RESPONSES_PATH) or ".", exist_ok=True)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_INTENT_RESPONSES, f, ensure_ascii=False, indent=2)
    event.log(f"[CONFIG] created default: {RESPONSES_PATH}")


class IntentResponder:
    """
    JSON 기반: intent -> (tts / beep / silent)
    """
    def __init__(self, config_path: str, speaker: TTSGuard, event: EventCSV):
        self.config_path = config_path
        self.speaker = speaker
        self.event = event
        self.cfg = {}
        self._load()

        self.last_intent = ""
        self.last_time = 0.0
        self.min_gap_sec = 0.25  # 같은 intent 연타 방지

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
        fmt = {"heard_text": heard_text}
        fmt.update(kwargs)

        if mode == "silent":
            return

        if mode == "beep":
            pattern = block.get("beep") or []
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
                pass
            self.event.log(f"[RESPOND] {intent} -> TTS")
            self.speaker.say(text, tag=f"{intent} TTS", force=force)
            return


# ======================
# Realtime Whisper (✅ TTS 중 mute)
# ======================
class RealtimeWhisper:
    def __init__(self, asr_model, sample_rate=16000, input_device=None, mute_fn=None):
        self.asr = asr_model
        self.sr = sample_rate
        self.device = input_device
        self.mute_fn = mute_fn or (lambda: False)

        self.q = queue.Queue()
        self.stop_event = threading.Event()

        self.chunk_frames = int(self.sr * (CHUNK_MS / 1000.0))
        self.min_speech_frames = int(self.sr * MIN_SPEECH_SEC)
        self.end_silence_frames = int(self.sr * END_SILENCE_SEC)
        self.max_utt_frames = int(self.sr * MAX_UTT_SEC)

        self._stream = None
        self.last_asr_ms = 0.0

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            print("[AUDIO]", status)
        self.q.put(indata[:, 0].copy())

    def start(self):
        self.stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_frames,
            device=self.device,
            callback=self._audio_cb,
        )
        self._stream.start()

    def stop(self):
        self.stop_event.set()
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def listen_texts(self):
        in_speech = False
        utt = np.zeros((0,), dtype=np.float32)
        silence = 0

        while not self.stop_event.is_set():
            # ✅ TTS 재생 중이면 ASR mute + 상태 리셋 + 큐 비우기
            if self.mute_fn():
                in_speech = False
                utt = np.zeros((0,), dtype=np.float32)
                silence = 0
                try:
                    while True:
                        self.q.get_nowait()
                except queue.Empty:
                    pass
                time.sleep(0.02)
                continue

            try:
                chunk = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
            if DEBUG_ENERGY:
                print(f"[RMS] {rms:.5f}")

            is_speech = rms > ENERGY_THRESHOLD

            if is_speech:
                in_speech = True
                silence = 0
                utt = np.concatenate([utt, chunk])

                if len(utt) > self.max_utt_frames:
                    text = self._transcribe_utt(utt)
                    if text:
                        yield text
                    utt = np.zeros((0,), dtype=np.float32)
                    in_speech = False
                    silence = 0
            else:
                if in_speech:
                    silence += len(chunk)
                    utt = np.concatenate([utt, chunk])
                    if silence >= self.end_silence_frames:
                        if len(utt) >= self.min_speech_frames:
                            text = self._transcribe_utt(utt)
                            if text:
                                yield text
                        utt = np.zeros((0,), dtype=np.float32)
                        in_speech = False
                        silence = 0

    def _transcribe_utt(self, audio_1d: np.ndarray) -> str:
        t0 = perf_counter()
        result = self.asr.transcribe(
            audio_1d,
            language="en",
            fp16=False,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=False,
        )
        self.last_asr_ms = (perf_counter() - t0) * 1000.0
        return (result.get("text") or "").strip()


# ======================
# Helpers
# ======================
def load_intent_bank(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def override_safety(text: str):
    t = text.lower()
    if any(k in t for k in PAIN_KEYWORDS):
        return "PAIN"
    if any(k in t for k in STOP_KEYWORDS):
        return "STOP"
    if any(k in t for k in START_KEYWORDS):
        return "START"
    return None

def extract_distance_m(text: str, default_m: float = 0.05) -> float:
    t = text.lower()
    m = re.search(r"(\d+(\.\d+)?)\s*(mm|millimeter|millimeters)", t)
    if m:
        return max(0.0, float(m.group(1)) / 1000.0)
    m = re.search(r"(\d+(\.\d+)?)\s*(cm|centimeter|centimeters)", t)
    if m:
        return max(0.0, float(m.group(1)) / 100.0)
    m = re.search(r"(\d+(\.\d+)?)\s*(m|meter|meters)", t)
    if m:
        return max(0.0, float(m.group(1)))
    return default_m

def parse_move_command(text: str):
    t = text.lower()
    found_dir = None
    for w in DIR_WORDS.keys():
        if re.search(rf"\b{re.escape(w)}\b", t):
            found_dir = w
            break
    if not found_dir:
        return None

    has_move_verb = any(re.search(rf"\b{re.escape(v)}\b", t) for v in MOVE_VERBS)
    if not has_move_verb:
        return None

    axis, sign = DIR_WORDS[found_dir]
    dist_m = extract_distance_m(t, default_m=0.05)

    dx = dy = dz = 0.0
    if axis == "x":
        dx = sign * dist_m
    elif axis == "y":
        dy = sign * dist_m
    else:
        dz = sign * dist_m
    return dx, dy, dz

def parse_sequence_command(text: str):
    t = text.lower()
    return bool(re.search(r"\b(start|run|begin)\b.*\b(sequence|scan|routine)\b", t))

def clamp_pose_xyz(pose):
    pose[0] = float(np.clip(pose[0], XYZ_LIMITS["x"][0], XYZ_LIMITS["x"][1]))
    pose[1] = float(np.clip(pose[1], XYZ_LIMITS["y"][0], XYZ_LIMITS["y"][1]))
    pose[2] = float(np.clip(pose[2], XYZ_LIMITS["z"][0], XYZ_LIMITS["z"][1]))
    return pose

def get_distance_xyz(p1, p2):
    return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2))

def top3_str(candidates):
    try:
        return " | ".join([f"{c['intent']}:{c['score']:.3f}" for c in candidates[:3]])
    except Exception:
        return ""


# ======================
# Motion worker (robot control only; NO speaking here)
# ======================
class MotionWorker:
    def __init__(self, rtde_c, rtde_r, event: EventCSV, state: dict):
        self.rtde_c = rtde_c
        self.rtde_r = rtde_r
        self.event = event
        self.state = state

        self.q = queue.Queue()
        self.stop_event = threading.Event()

        self.control_lock = threading.Lock()
        self.control = None  # "STOP" or "PAIN"
        self.control_event = threading.Event()

        self.total_moved = 0.0
        self.last_pose = None
        self.last_brief_time = time.time()

        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.stop_event.clear()
        self.thread.start()

    def shutdown(self):
        self.stop_event.set()
        self.control_event.set()
        try:
            self.q.put_nowait({"type": "noop"})
        except Exception:
            pass

    def request_move_delta(self, dx, dy, dz, speed, acc, utt_id=None, text=None):
        self.q.put({
            "type": "move_delta",
            "dx": float(dx), "dy": float(dy), "dz": float(dz),
            "speed": float(speed), "acc": float(acc),
            "utt_id": utt_id, "text": text
        })

    def request_sequence_down(self, steps=100, step_dz=-0.005, speed=None, acc=None, utt_id=None, text=None):
        self.q.put({
            "type": "sequence_down",
            "steps": int(steps),
            "step_dz": float(step_dz),
            "speed": float(speed if speed is not None else BASE_SPEED),
            "acc": float(acc if acc is not None else BASE_ACC),
            "utt_id": utt_id, "text": text
        })

    def trigger_control(self, kind: str):
        with self.control_lock:
            self.control = kind
        self.control_event.set()

    def _clear_queue(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            return

    def _safe_stop(self):
        try:
            self.rtde_c.stopL(2.0)
            return True
        except Exception:
            try:
                self.rtde_c.stopJ(2.0)
                return True
            except Exception:
                return False

    def _handle_control_if_any(self):
        if not self.control_event.is_set():
            return False

        with self.control_lock:
            kind = self.control
            self.control = None
        self.control_event.clear()

        if kind == "STOP":
            self.event.log("비상 정지 트리거(STOP)")
            self._safe_stop()
            self.state["armed"] = False
            self._clear_queue()
            return True

        if kind == "PAIN":
            self.event.log("PAIN 트리거")
            self._safe_stop()
            time.sleep(0.05)
            try:
                pose = self.rtde_r.getActualTCPPose()
                if self.last_pose is None:
                    self.last_pose = pose

                target = pose.copy()
                target[2] += PAIN_RETRACT_Z_M
                target = clamp_pose_xyz(target)

                if not DRY_RUN:
                    self.rtde_c.moveL(target, BASE_SPEED * 0.5, BASE_ACC * 0.5, True)
                    self._poll_until_reached(target)
                self.event.log("PAIN retract 완료")
            except Exception as e:
                self.event.log(f"PAIN retract 실패: {repr(e)}")

            self.state["armed"] = False
            self._clear_queue()
            return True

        return False

    def _poll_until_reached(self, target_pose):
        while not self.stop_event.is_set():
            if self._handle_control_if_any():
                return False

            curr = self.rtde_r.getActualTCPPose()
            if self.last_pose is None:
                self.last_pose = curr

            step = get_distance_xyz(self.last_pose, curr)
            self.total_moved += step
            self.last_pose = curr

            # (기본 OFF) 주기 브리핑은 여기서 하지 않음
            if get_distance_xyz(curr, target_pose) < POSE_TOL_M:
                return True

            time.sleep(0.02)

        return False

    def _do_move_delta(self, cmd):
        dx, dy, dz = cmd["dx"], cmd["dy"], cmd["dz"]
        speed, acc = cmd["speed"], cmd["acc"]

        pose = self.rtde_r.getActualTCPPose()
        if self.last_pose is None:
            self.last_pose = pose

        target = pose.copy()
        target[0] += dx
        target[1] += dy
        target[2] += dz
        target = clamp_pose_xyz(target)

        self.event.log(f"moveL async start dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}")

        if DRY_RUN:
            return

        self.rtde_c.moveL(target, speed, acc, True)
        self._poll_until_reached(target)

    def _do_sequence_down(self, cmd):
        steps = cmd["steps"]
        step_dz = cmd["step_dz"]
        speed = cmd["speed"] * self.state.get("speed_scale", 1.0)
        acc = cmd["acc"] * self.state.get("speed_scale", 1.0)

        self.event.log(f"sequence_down start steps={steps} step_dz={step_dz}")

        pose = self.rtde_r.getActualTCPPose()
        if self.last_pose is None:
            self.last_pose = pose
        target = list(pose)

        for _ in range(steps):
            if self._handle_control_if_any():
                return
            target[2] += step_dz
            target = clamp_pose_xyz(target)
            if DRY_RUN:
                time.sleep(0.02)
                continue
            self.rtde_c.moveL(target, speed, acc, True)
            ok = self._poll_until_reached(target)
            if not ok:
                return

        self.event.log("sequence_down done")

    def _run(self):
        while not self.stop_event.is_set():
            if self._handle_control_if_any():
                continue
            try:
                cmd = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._handle_control_if_any():
                continue

            ctype = cmd.get("type")
            try:
                if ctype == "move_delta":
                    self._do_move_delta(cmd)
                elif ctype == "sequence_down":
                    self._do_sequence_down(cmd)
            except Exception as e:
                self.event.log(f"[MOTION ERROR] {repr(e)}")


# ======================
# main
# ======================
def main():
    metrics = None
    event = None
    listener = None
    rtde_c = None
    rtde_r = None
    motion = None
    briefing = None

    try:
        metrics_path = os.path.join("logs", f"metrics_{datetime.now():%Y%m%d_%H%M%S}.csv")
        event_path = os.path.join("logs", f"event_log_{datetime.now():%Y%m%d_%H%M%S}.csv")
        metrics = MetricsCSV(metrics_path)
        event = EventCSV(event_path)

        event.log(f"event_log created: {event_path}")

        # TTS
        briefing = BriefingWrapper(event)
        speaker = TTSGuard(briefing)

        # ensure response json exists + load responder
        ensure_intent_responses_file(event)
        responder = IntentResponder(RESPONSES_PATH, speaker, event)

        if ENABLE_STARTUP_TTS:
            speaker.say("Audio output ready.", tag="STARTUP", force=True)

        # Whisper
        event.log(f"[ASR] Loading Whisper: {WHISPER_MODEL}")
        t0 = perf_counter()
        asr = whisper.load_model(WHISPER_MODEL)
        metrics.log(kind="startup", step="load_whisper", duration_ms=f"{(perf_counter()-t0)*1000.0:.3f}", note=WHISPER_MODEL)

        # MiniLM
        event.log("[NLU] Loading intent bank + MiniLM...")
        intent_bank = load_intent_bank(INTENT_BANK_PATH)
        retriever = MiniLMRetriever(intent_bank)
        router = IntentRouter(retriever, threshold=THRESHOLD)

        # RTDE
        event.log(f"[RTDE] Connecting to: {HOST}")
        rtde_c = rtde_control.RTDEControlInterface(HOST)
        rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
        event.log("[RTDE] Connected")

        state = {"armed": False, "speed_scale": 1.0}

        # motion worker
        motion = MotionWorker(rtde_c, rtde_r, event, state)
        motion.start()

        # ✅ ASR mute while TTS speaking (feedback-loop prevention)
        listener = RealtimeWhisper(
            asr,
            sample_rate=SAMPLE_RATE,
            input_device=INPUT_DEVICE,
            mute_fn=lambda: briefing.is_speaking
        )
        listener.start()

        print("\nRealtime listening... (Ctrl+C to quit)\n"
              "- Say 'start' to arm\n"
              "- Say 'stop' to emergency stop (beep)\n"
              "- Say 'it hurts' / 'painful' to trigger PAIN(stop + retract + TTS)\n"
              "- Try: 'move up 5 cm', 'move forward 10 centimeters'\n"
              "- Optional: 'start sequence'\n")

        utt_id = 0

        for text in listener.listen_texts():
            utt_id += 1
            asr_ms = float(getattr(listener, "last_asr_ms", 0.0) or 0.0)

            print("[TEXT]", text if text else "(empty)")
            if not text:
                continue

            # 1) ultra-fast safety override
            ov = override_safety(text)
            if ov == "STOP":
                motion.trigger_control("STOP")
                responder.respond("STOP", heard_text=text, force=True)  # ✅ beep
                continue

            if ov == "PAIN":
                motion.trigger_control("PAIN")
                responder.respond("PAIN", heard_text=text, force=True)  # ✅ sentence TTS
                continue

            if ov == "START":
                state["armed"] = True
                event.log("ARMED = True (START)")
                responder.respond("START", heard_text=text, force=True)
                continue

            # 2) optional sequence command
            if parse_sequence_command(text):
                if not state["armed"]:
                    if SPEAK_BLOCKED_NOT_ARMED:
                        speaker.say("Not armed. Say start first.", tag="NOT_ARMED")
                    continue
                speed = BASE_SPEED * state["speed_scale"]
                acc = BASE_ACC * state["speed_scale"]
                motion.request_sequence_down(steps=100, step_dz=-0.005, speed=speed, acc=acc, utt_id=utt_id, text=text)
                responder.respond("SEQUENCE", heard_text=text, force=True)
                continue

            # 3) rule-based move
            delta = parse_move_command(text)
            if delta:
                if not state["armed"]:
                    if SPEAK_BLOCKED_NOT_ARMED:
                        speaker.say("Not armed. Say start first.", tag="NOT_ARMED")
                    continue
                dx, dy, dz = delta
                speed = BASE_SPEED * state["speed_scale"]
                acc = BASE_ACC * state["speed_scale"]
                motion.request_move_delta(dx, dy, dz, speed, acc, utt_id=utt_id, text=text)
                responder.respond("RULE_MOVE", heard_text=text)  # default: silent
                continue

            # 4) MiniLM routing
            chosen, candidates = router.route(text)
            if not chosen:
                continue

            intent = (chosen.get("intent") or "").upper()
            score = chosen.get("score", "")

            # safety intents from router too
            if intent == "STOP":
                motion.trigger_control("STOP")
                responder.respond("STOP", heard_text=text, force=True)
                continue

            if intent == "PAIN":
                motion.trigger_control("PAIN")
                responder.respond("PAIN", heard_text=text, force=True)
                continue

            if intent == "START":
                state["armed"] = True
                responder.respond("START", heard_text=text, force=True)
                continue

            if not state["armed"]:
                if SPEAK_BLOCKED_NOT_ARMED:
                    speaker.say("Not armed. Say start first.", tag="NOT_ARMED")
                continue

            if intent == "MOVE_SLOW":
                state["speed_scale"] = 0.5
                responder.respond("MOVE_SLOW", heard_text=text)
                continue

            if intent == "OK":
                state["speed_scale"] = 1.0
                responder.respond("OK", heard_text=text)
                continue

            if intent == "EXPLAIN":
                responder.respond("EXPLAIN", heard_text=text)
                continue

            # unhandled
            if SPEAK_UNHANDLED_INTENT:
                speaker.say(f"Unhandled intent: {intent}", tag="UNHANDLED")

    except KeyboardInterrupt:
        print("\n[QUIT] Ctrl+C")
    finally:
        if listener:
            listener.stop()
        if motion:
            motion.shutdown()
        if briefing:
            try:
                briefing.stop()
            except Exception:
                pass
        if rtde_c:
            try:
                rtde_c.stopScript()
            except Exception:
                pass
            try:
                rtde_c.disconnect()
            except Exception:
                pass
        if rtde_r:
            try:
                rtde_r.disconnect()
            except Exception:
                pass
        if metrics:
            metrics.close()
        if event:
            event.log("shutdown complete")
            event.close()
        print("[RTDE] Closed")


if __name__ == "__main__":
    main()
