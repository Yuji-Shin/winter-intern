# final.py
# Realtime Whisper (utterance-level) + MiniLM intent routing + UR RTDE control
# + CSV metrics logging (startup + per-utterance timings)

import os
import csv
import json
import re
import time
import queue
import threading
from datetime import datetime
from time import perf_counter

import numpy as np
import sounddevice as sd
import whisper

import rtde_control
import rtde_receive

from src.minilm import MiniLMRetriever
from src.router import IntentRouter


# ======================
# Audio / ASR settings
# ======================
SAMPLE_RATE = 16000
INPUT_DEVICE = 1            # 문제 있으면 None(기본 장치)로 바꿔보기
WHISPER_MODEL = "base.en"   # 느리면 "tiny.en" 추천

# Realtime chunk / (simple) VAD-ish settings
CHUNK_MS = 30               # 20~50ms 권장
ENERGY_THRESHOLD = 0.012    # 작게=민감(잡음도 speech), 크게=둔감(말 놓침)
MIN_SPEECH_SEC = 0.25       # 이보다 짧으면 발화로 안 침
END_SILENCE_SEC = 0.55      # 이만큼 조용하면 발화 종료로 판단
MAX_UTT_SEC = 4.5           # 발화 최대 길이(지연 방지)
DEBUG_ENERGY = False        # True면 RMS 출력(튜닝할 때만)


# ======================
# NLU (MiniLM) settings
# ======================
INTENT_BANK_PATH = "data/intent_bank_en.json"
THRESHOLD = 0.55


# ======================
# Robot settings
# ======================
HOST = "127.0.0.1"     # URSim IP (VM이면 VM IP로 변경)
BASE_SPEED = 0.25
BASE_ACC = 0.5
DRY_RUN = False


# ======================
# Safety / behavior
# ======================
PAIN_RETRACT_Z_M = 0.03  # 3cm

XYZ_LIMITS = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (0.0, 1.2),
}

STOP_KEYWORDS  = ["stop", "pause", "hold on", "wait", "halt", "freeze"]
START_KEYWORDS = ["start", "begin", "continue", "go", "resume", "arm"]

# (선택) PAIN 키워드 오버라이드(초저지연 대응 원하면 켜기)
PAIN_KEYWORDS = ["it hurts", "hurt", "pain", "ouch"]

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
# CSV Metrics Logger
# ======================
def _now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


class MetricsCSV:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.fields = [
            "ts",
            "kind",          # startup / utterance
            "step",          # load_whisper, init_minilm, route, moveL...
            "utt_id",
            "duration_ms",
            "asr_ms",
            "text",
            "override_intent",
            "chosen_intent",
            "chosen_score",
            "top3",
            "note",
        ]
        self.w = csv.DictWriter(self.f, fieldnames=self.fields)
        self.w.writeheader()
        self.f.flush()

    def log(self, **row):
        base = {k: "" for k in self.fields}
        base.update(row)
        base["ts"] = base["ts"] or _now_iso()
        self.w.writerow(base)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# ======================
# Realtime Whisper helper
# ======================
class RealtimeWhisper:
    """
    마이크 스트림을 계속 받으면서,
    에너지 기반으로 발화를 잘라 Whisper에 넣고 text를 yield.
    (토큰 스트리밍은 아니고, '발화 단위'로 거의 실시간처럼 동작)
    """
    def __init__(self, asr_model, sample_rate=16000, input_device=None):
        self.asr = asr_model
        self.sr = sample_rate
        self.device = input_device

        self.q = queue.Queue()
        self.stop_event = threading.Event()

        self.chunk_frames = int(self.sr * (CHUNK_MS / 1000.0))
        self.min_speech_frames = int(self.sr * MIN_SPEECH_SEC)
        self.end_silence_frames = int(self.sr * END_SILENCE_SEC)
        self.max_utt_frames = int(self.sr * MAX_UTT_SEC)

        self._stream = None

        # last utterance ASR time (ms)
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

    # 초저지연 PAIN(원치 않으면 이 블록 삭제해도 됨)
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
    """
    "move up 5 cm" 같은 규칙 기반 이동 명령 파싱.
    성공하면 (dx,dy,dz) 반환, 아니면 None.
    """
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


def clamp_pose_xyz(pose):
    pose[0] = float(np.clip(pose[0], XYZ_LIMITS["x"][0], XYZ_LIMITS["x"][1]))
    pose[1] = float(np.clip(pose[1], XYZ_LIMITS["y"][0], XYZ_LIMITS["y"][1]))
    pose[2] = float(np.clip(pose[2], XYZ_LIMITS["z"][0], XYZ_LIMITS["z"][1]))
    return pose


def safe_stop(rtde_c):
    try:
        rtde_c.stopL(2.0)
        return True
    except Exception:
        try:
            rtde_c.stopJ(2.0)
            return True
        except Exception:
            return False


def moveL_delta(rtde_c, rtde_r, dx, dy, dz, speed, acc):
    pose = rtde_r.getActualTCPPose()
    print("current TCP pose:", pose)

    target = pose.copy()
    target[0] += float(dx)
    target[1] += float(dy)
    target[2] += float(dz)
    target = clamp_pose_xyz(target)

    print("target  TCP pose:", target)

    if DRY_RUN:
        print("[DRY_RUN] skip moveL")
        return True

    ok = rtde_c.moveL(target, speed, acc)
    return ok


def handle_intent(intent: str, rtde_c, rtde_r, state):
    """
    intent -> 로봇 동작/상태 변경
    state: dict (armed, speed_scale 등)
    """
    intent = (intent or "").upper()

    if intent == "STOP":
        print("[INTENT] STOP -> stopping + disarming")
        safe_stop(rtde_c)
        state["armed"] = False
        return

    if intent == "START":
        state["armed"] = True
        print("[INTENT] START -> ARMED=True")
        return

    if intent == "PAIN":
        print("[INTENT] PAIN -> stop + retract(up) + disarm")
        safe_stop(rtde_c)
        time.sleep(0.1)
        retract_speed = BASE_SPEED * 0.5
        retract_acc = BASE_ACC * 0.5
        ok = moveL_delta(rtde_c, rtde_r, 0.0, 0.0, +PAIN_RETRACT_Z_M, retract_speed, retract_acc)
        print("retract ok:", ok)
        state["armed"] = False
        return

    if intent == "MOVE_SLOW":
        state["speed_scale"] = 0.5
        print("[INTENT] MOVE_SLOW -> speed_scale=0.5")
        return

    if intent == "OK":
        state["speed_scale"] = 1.0
        print("[INTENT] OK -> speed_scale=1.0")
        return

    if intent == "EXPLAIN":
        print("[INTENT] EXPLAIN -> (print) I am moving the tool based on your commands.")
        return

    print("[INTENT] Unknown/Unhandled:", intent)


def top3_str(candidates):
    try:
        return " | ".join([f"{c['intent']}:{c['score']:.3f}" for c in candidates[:3]])
    except Exception:
        return ""


# ======================
# main (REALTIME)
# ======================
def main():
    metrics = None
    listener = None
    rtde_c = None
    rtde_r = None

    try:
        metrics_path = os.path.join("logs", f"metrics_{datetime.now():%Y%m%d_%H%M%S}.csv")
        metrics = MetricsCSV(metrics_path)
        metrics.log(kind="startup", step="start", duration_ms=0, note=f"csv={metrics_path}")

        print("[ASR] Loading Whisper:", WHISPER_MODEL)
        t0 = perf_counter()
        asr = whisper.load_model(WHISPER_MODEL)
        metrics.log(kind="startup", step="load_whisper", duration_ms=(perf_counter() - t0) * 1000.0, note=WHISPER_MODEL)

        print("[NLU] Loading intent bank + MiniLM retriever/router...")
        t0 = perf_counter()
        intent_bank = load_intent_bank(INTENT_BANK_PATH)
        metrics.log(kind="startup", step="load_intent_bank", duration_ms=(perf_counter() - t0) * 1000.0, note=INTENT_BANK_PATH)

        t0 = perf_counter()
        retriever = MiniLMRetriever(intent_bank)
        router = IntentRouter(retriever, threshold=THRESHOLD)
        metrics.log(kind="startup", step="init_minilm_router", duration_ms=(perf_counter() - t0) * 1000.0, note=f"threshold={THRESHOLD}")

        print("[RTDE] Connecting to:", HOST)
        t0 = perf_counter()
        rtde_c = rtde_control.RTDEControlInterface(HOST)
        rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
        metrics.log(kind="startup", step="connect_rtde", duration_ms=(perf_counter() - t0) * 1000.0, note=HOST)

        state = {"armed": False, "speed_scale": 1.0}

        print("\nRealtime listening... (Ctrl+C to quit)\n"
              "- Say 'start' to arm\n"
              "- Say 'stop' to stop+disarm\n"
              "- Say 'it hurts' to trigger PAIN behavior\n"
              "- Try: 'move up 5 cm', 'move forward 10 centimeters'\n")

        listener = RealtimeWhisper(asr, sample_rate=SAMPLE_RATE, input_device=INPUT_DEVICE)
        listener.start()

        utt_id = 0

        for text in listener.listen_texts():
            utt_id += 1
            asr_ms = float(getattr(listener, "last_asr_ms", 0.0) or 0.0)

            print("[TEXT]", text if text else "(empty)")
            if not text:
                metrics.log(kind="utterance", step="empty_text", utt_id=utt_id, duration_ms=0, asr_ms=asr_ms, text="")
                continue

            # 1) 안전 키워드 override 최우선(초저지연)
            t0 = perf_counter()
            ov = override_safety(text)
            ov_ms = (perf_counter() - t0) * 1000.0

            if ov:
                t1 = perf_counter()
                handle_intent(ov, rtde_c, rtde_r, state)
                act_ms = (perf_counter() - t1) * 1000.0

                metrics.log(
                    kind="utterance",
                    step="override",
                    utt_id=utt_id,
                    duration_ms=f"{ov_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    override_intent=ov,
                    chosen_intent=ov,
                    note=f"action_ms={act_ms:.3f}, armed={state['armed']}, speed_scale={state['speed_scale']}"
                )
                print("[STATE]", state, "\n")
                continue

            # 2) 규칙 기반 move 파싱이 되면 그걸 먼저 실행
            t0 = perf_counter()
            delta = parse_move_command(text)
            parse_ms = (perf_counter() - t0) * 1000.0

            if delta:
                if not state["armed"]:
                    metrics.log(
                        kind="utterance",
                        step="rule_move_blocked_not_armed",
                        utt_id=utt_id,
                        duration_ms=f"{parse_ms:.3f}",
                        asr_ms=f"{asr_ms:.3f}",
                        text=text,
                        chosen_intent="RULE_MOVE",
                        note="blocked:not_armed"
                    )
                    print("[BLOCKED] Not armed. Say 'start' first.\n")
                    continue

                dx, dy, dz = delta
                speed = BASE_SPEED * state["speed_scale"]
                acc = BASE_ACC * state["speed_scale"]
                print(f"[RULE_MOVE] dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}  speed={speed:.3f}")

                t1 = perf_counter()
                ok = moveL_delta(rtde_c, rtde_r, dx, dy, dz, speed, acc)
                move_ms = (perf_counter() - t1) * 1000.0

                metrics.log(
                    kind="utterance",
                    step="rule_move",
                    utt_id=utt_id,
                    duration_ms=f"{move_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    chosen_intent="RULE_MOVE",
                    note=f"parse_ms={parse_ms:.3f}, ok={ok}, dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, speed={speed:.3f}"
                )

                print("moveL ok:", ok, "\n")
                continue

            # 3) MiniLM intent 라우팅
            t0 = perf_counter()
            chosen, candidates = router.route(text)
            route_ms = (perf_counter() - t0) * 1000.0
            top3 = top3_str(candidates)

            print("\n[Top-3 candidates]")
            for c in candidates[:3]:
                print(f"- {c['intent']:10s} score={c['score']:.3f} ({c['text']})")

            if not chosen:
                metrics.log(
                    kind="utterance",
                    step="route_no_intent",
                    utt_id=utt_id,
                    duration_ms=f"{route_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    top3=top3,
                    note="below_threshold"
                )
                print("\n[NO_INTENT] below threshold\n")
                continue

            intent = chosen["intent"]
            score = chosen.get("score", "")

            print("\n[CHOSEN]", chosen)

            # PAIN/STOP/START 같은 안전 intent는 armed 여부와 관계 없이 처리
            if intent in ["PAIN", "STOP", "START"]:
                t1 = perf_counter()
                handle_intent(intent, rtde_c, rtde_r, state)
                act_ms = (perf_counter() - t1) * 1000.0

                metrics.log(
                    kind="utterance",
                    step="route_intent_safety",
                    utt_id=utt_id,
                    duration_ms=f"{route_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    chosen_intent=intent,
                    chosen_score=score,
                    top3=top3,
                    note=f"action_ms={act_ms:.3f}, armed={state['armed']}, speed_scale={state['speed_scale']}"
                )

                print("[STATE]", state, "\n")
                continue

            # 나머지는 start(armed) 이후에만 실행
            if not state["armed"]:
                metrics.log(
                    kind="utterance",
                    step="route_intent_blocked_not_armed",
                    utt_id=utt_id,
                    duration_ms=f"{route_ms:.3f}",
                    asr_ms=f"{asr_ms:.3f}",
                    text=text,
                    chosen_intent=intent,
                    chosen_score=score,
                    top3=top3,
                    note="blocked:not_armed"
                )
                print("[BLOCKED] Not armed. Say 'start' first.\n")
                continue

            t1 = perf_counter()
            handle_intent(intent, rtde_c, rtde_r, state)
            act_ms = (perf_counter() - t1) * 1000.0

            metrics.log(
                kind="utterance",
                step="route_intent",
                utt_id=utt_id,
                duration_ms=f"{route_ms:.3f}",
                asr_ms=f"{asr_ms:.3f}",
                text=text,
                chosen_intent=intent,
                chosen_score=score,
                top3=top3,
                note=f"action_ms={act_ms:.3f}, armed={state['armed']}, speed_scale={state['speed_scale']}"
            )

            print("[STATE]", state, "\n")

    except KeyboardInterrupt:
        print("\n[QUIT] Ctrl+C")
        if metrics:
            metrics.log(kind="startup", step="keyboard_interrupt", duration_ms=0)
    except Exception as e:
        print("\n[ERROR]", repr(e))
        if metrics:
            metrics.log(kind="startup", step="exception", duration_ms=0, note=repr(e))
        raise
    finally:
        if listener:
            listener.stop()
        if rtde_c:
            try:
                rtde_c.stopScript()
            except Exception:
                pass
        if metrics:
            metrics.log(kind="startup", step="end", duration_ms=0)
            metrics.close()
        print("[RTDE] Closed")


if __name__ == "__main__":
    main()
