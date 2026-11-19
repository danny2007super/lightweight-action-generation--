"""Continuous, supervised ACT rollout for the physical LEFT Dobot arm.

Press S in the camera window to start, R to drive the arm back to the pose that
was recorded when the last run started, Q or Esc to quit. No step limit, no time
limit, no action shaping.

ACT deployment follows the reference implementation: the policy is queried every
control tick, each query returns a 30-step action chunk, and the action executed
now is the exponentially weighted average over every chunk that still covers the
current time index. Running one chunk to completion open loop and only then
re-planning makes the arm re-enter the same approach over and over, because each
new plan starts from a state the previous plan did not expect.
"""
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PORTABLE_SITE = Path(os.environ.get("DOBOT_PORTABLE_SITE", str(HERE.parent / "optional_site_packages")))
ROBOT_REPO = HERE.parent / "robot_bridge"
FLOW_DIR = HERE.parent / "FlowMatching准备"
sys.path.append(str(PORTABLE_SITE))
sys.path.insert(0, str(ROBOT_REPO))

# "act" loads the ACT checkpoint, "fm" the flow-matching one. Both expose
# predict(left_rgb, top_rgb, qpos) -> {"left_arm_7": (30, 7)}. They must never be
# imported into the same process: both ship a top-level module called "model".
POLICY_NAME = os.environ.get("ACT_POLICY", "act").strip().lower()
if POLICY_NAME not in ("act", "fm"):
    raise SystemExit(f"ACT_POLICY must be 'act' or 'fm', got {POLICY_NAME!r}")

import cv2  # noqa: E402
if POLICY_NAME == "fm":
    sys.path.insert(0, str(FLOW_DIR))
    from flow_inference import FlowInference as PolicyClass  # noqa: E402
else:
    from act_inference import ACTInference as PolicyClass  # noqa: E402
from dobot_control.cameras.realsense_camera import RealSenseCamera  # noqa: E402
from dobot_control.robots.dobot_api import DobotApiDashboard, DobotApiMove  # noqa: E402
from third_party.feetech.scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS  # noqa: E402

LEFT_IP = os.environ.get("DOBOT_LEFT_IP", "192.168.5.2")
RIGHT_IP = os.environ.get("DOBOT_RIGHT_IP", "192.168.5.1")
CONTROL_HZ = 15.0
SERVO_T = 0.05  # ServoJ interpolation window, close to the 1/15 s control period
SPEED_PERCENT = 50  # ceiling only; the 15 Hz plan itself asks for about 0.16 rad/s
ENSEMBLE_DECAY = 0.06 if POLICY_NAME == "fm" else 0.1  # fm: longer window, less jitter
EMA_ALPHA = 0.7  # smoothing of the joint target; 1.0 disables it
FLOW_SEED = int(os.environ.get("FLOW_SEED", "0"))  # fixed noise => deterministic chunk
FM_SMOOTH_PASSES = int(os.environ.get("FM_SMOOTH", "2"))  # [1,2,1] passes on the chunk
GRIPPER_OPEN = 1.0
GRIPPER_SPEED = 3000  # steps/s; demo teleop used the servo maximum (~4096)
GRASP_HOLD_S = 0.4  # hold the arm still while the jaws finish a big move
GRASP_HOLD_TRIGGER = 0.08  # gripper target jump that counts as a grasp/release
AUTO_STOP_AFTER_RELEASE = os.environ.get("AUTO_STOP", "1") != "0"
RELEASE_HOLD_S = 2.0  # after the tube is released, run this long then stop
POSE_PATH = HERE / "act_start_pose.json"
RETURN_SPEED_PERCENT = 10  # joint move back to the recorded pose is deliberately slow
RETURN_TIMEOUT_S = 20.0


def checked(reply, name):
    if not str(reply).startswith("0,"):
        raise RuntimeError(f"{name}: {reply!r}")
    return reply


def joints(api):
    reply = checked(api.GetAngle(), "GetAngle")
    match = re.search(r"\{([^}]*)\}", reply)
    if not match:
        raise RuntimeError(f"Invalid GetAngle: {reply!r}")
    value = np.deg2rad(np.array([float(v) for v in match.group(1).split(",")]))
    if value.shape != (6,) or not np.isfinite(value).all():
        raise RuntimeError("Invalid joint feedback")
    return value


def mode(api):
    reply = checked(api.RobotMode(), "RobotMode")
    match = re.search(r"\{(\d+)\}", reply)
    if not match:
        raise RuntimeError(f"Invalid RobotMode: {reply!r}")
    return int(match.group(1))


def grip_read(servo):
    raw, status, error = servo.ReadPos(21)
    if status != COMM_SUCCESS or error:
        raise RuntimeError(f"Left gripper feedback failed: {status}, {error}")
    value = (raw - 2048) / 1004
    if not -0.01 <= value <= 1.01:
        raise RuntimeError(f"Unexpected gripper position: {value}")
    return float(np.clip(value, 0, 1))


def grip_write(servo, value):
    goal = round(2048 + float(np.clip(value, 0, 1)) * 1004)
    # 100 steps/s (the old value) needed about 10 s for a full stroke; the teleop
    # demonstrations drove the same servo at 4096, so the policy expects the jaws
    # to close in roughly 0.25 s and starts lifting long before that.
    status, error = servo.WritePosEx(21, goal, GRIPPER_SPEED, 0)
    if status != COMM_SUCCESS or error:
        raise RuntimeError(f"Gripper command failed: {status}, {error}")
    return goal


def smooth_chunk(chunk, passes=2):
    """Light [1,2,1] smoothing along the chunk's time axis.

    The flow checkpoint plans motion that reverses direction almost every frame
    (|ddq|/|dq| about 1.2-1.6, ACT is 0.44), which reads as stop-and-go even with
    perfectly regular servo timing. Two passes bring it to ACT-level smoothness.
    """
    out = np.asarray(chunk, dtype=np.float64)
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return out.astype(np.float32)


class CameraReader(threading.Thread):
    """Keeps the newest frame of one RealSense so the control tick never blocks.

    RealSenseCamera.read() waits for the next hardware frame, up to 1/fps. With two
    unsynchronised cameras read back to back, one missed frame doubled the control
    tick to ~140 ms, which is exactly where the arm appeared to stall.
    """

    def __init__(self, camera, name):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.camera = camera
        self.name = name
        self.lock = threading.Lock()
        self.image = None
        self.stamp = 0.0
        self.error = None
        self.running = True

    def run(self):
        while self.running:
            try:
                image, _ = self.camera.read()
            except Exception as exc:  # surfaced on the next latest() call
                self.error = exc
                return
            with self.lock:
                self.image = image
                self.stamp = time.monotonic()

    def latest(self):
        if self.error is not None:
            raise RuntimeError(f"camera {self.name} failed: {self.error!r}")
        with self.lock:
            return self.image, self.stamp

    def stop(self):
        self.running = False


def frames_from(readers):
    left, left_stamp = readers["left"].latest()
    top, top_stamp = readers["top"].latest()
    if left is None or top is None:
        raise RuntimeError("Camera frames not ready")
    if abs(left_stamp - top_stamp) > 0.2:
        raise RuntimeError("Camera frames too far apart")
    return left, top


def frame(left_cam, top_cam):
    left, _ = left_cam.read()
    t1 = time.monotonic()
    top, _ = top_cam.read()
    t2 = time.monotonic()
    if t2 - t1 > 0.1:
        raise RuntimeError("Camera frames too far apart")
    return left, top


def show(left, top, heading):
    small_left = cv2.resize(left[:, :, ::-1], (640, 480))
    small_top = cv2.resize(top[:, :, ::-1], (640, 480))
    canvas = np.concatenate((small_left, small_top), axis=1)
    cv2.putText(canvas, heading, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.imshow("ACT live: left | top", canvas)
    return cv2.waitKey(1) & 0xFF


def load_pose():
    try:
        data = json.loads(POSE_PATH.read_text(encoding="utf-8"))
        arm = np.array(data["left_arm_rad"], dtype=float)
    except Exception:
        return None
    if arm.shape != (6,) or not np.isfinite(arm).all():
        return None
    return arm


def save_pose(arm):
    POSE_PATH.write_text(json.dumps({"left_arm_rad": [float(v) for v in arm],
                                     "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                                    ensure_ascii=False, indent=2), encoding="utf-8")


def alarm_codes(api):
    try:
        reply = str(api.GetErrorID())
    except Exception:
        return []
    if "{" not in reply or "}" not in reply:
        return []
    return [int(v) for v in re.findall(r"\d+", reply.split("{", 1)[1].split("}", 1)[0])]


def ensure_ready(left, log):
    """Clear a latched controller alarm; mode 9 refuses enable and motion."""
    current = mode(left)
    if current in (4, 5):
        return current
    codes = alarm_codes(left)
    print(f"Left arm mode {current} (alarm {codes}); sending ClearError().", flush=True)
    checked(left.ClearError(), "ClearError")
    log["cleared_alarms"] = codes
    time.sleep(1.0)
    current = mode(left)
    if current not in (4, 5):
        raise RuntimeError(f"Left arm still in mode {current} after ClearError; "
                           "reset it on the teach pendant")
    return current


def return_to_start(left, motion, target, readers, log):
    """Move the arm back to a recorded joint pose at reduced speed. Q aborts mid move."""
    if mode(left) == 7:
        print("Arm is still running; press Q on a run before returning.", flush=True)
        return False
    ensure_ready(left, log)
    if mode(left) == 4:
        checked(left.EnableRobot(), "EnableRobot")
    if mode(left) != 5:
        print(f"Arm mode {mode(left)} is not ready for a return move.", flush=True)
        return False
    checked(left.SpeedFactor(RETURN_SPEED_PERCENT), "SpeedFactor")
    checked(left.SpeedJ(RETURN_SPEED_PERCENT), "SpeedJ")
    checked(left.AccJ(RETURN_SPEED_PERCENT), "AccJ")
    reply = motion.JointMovJ(*np.rad2deg(target).tolist())
    log["return_reply"] = str(reply)
    if reply and not str(reply).startswith("0,"):
        raise RuntimeError(f"MovJ rejected: {reply!r}")
    deadline = time.monotonic() + RETURN_TIMEOUT_S
    error = float("inf")
    while time.monotonic() < deadline:
        im_left, im_top = frames_from(readers)
        error = float(np.abs(joints(left) - target).max())
        key = show(im_left, im_top, f"RETURNING  err {error:.2f} rad  Q=ABORT")
        if key in (ord("q"), 27):
            left.DisableRobot()
            log["return_error_rad"] = error
            log["return_aborted"] = True
            print("Return aborted.", flush=True)
            return False
        if error < 0.01:
            break
        time.sleep(0.05)
    left.DisableRobot()
    log["return_error_rad"] = error
    return error < 0.01


def main():
    log_path = HERE / f"{POLICY_NAME}_live_steps.jsonl"
    summary_path = HERE / f"{POLICY_NAME}_live_summary.json"
    policy = PolicyClass()
    cameras = {}
    readers = {}
    right = left = motion = None
    port = None
    enabled_here = False
    log = {"mode": "temporal_ensemble_per_tick", "policy": POLICY_NAME,
           "run_dir": str(getattr(policy, "run_dir", "")),
           "checkpoint_update": policy.checkpoint_update,
           "steps_completed": 0, "inferences": 0,
           "settings": {"control_hz": CONTROL_HZ, "servo_t": SERVO_T,
                        "speed_percent": SPEED_PERCENT,
                        "ensemble_decay": ENSEMBLE_DECAY,
                        "ema_alpha": EMA_ALPHA,
                        "flow_seed": FLOW_SEED if POLICY_NAME == "fm" else None,
                        "shaping": "EMA-smoothed weighted mean of overlapping chunks"}}
    stream = log_path.open("w", encoding="utf-8", buffering=1)
    try:
        cameras["left"] = RealSenseCamera(os.environ["ACT_LEFT_CAMERA_SERIAL"], flip=True, fps=15)
        cameras["top"] = RealSenseCamera(os.environ["ACT_TOP_CAMERA_SERIAL"], flip=True, fps=15)
        readers = {name: CameraReader(camera, name) for name, camera in cameras.items()}
        for reader in readers.values():
            reader.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if all(reader.latest()[0] is not None for reader in readers.values()):
                break
            time.sleep(0.05)
        right = DobotApiDashboard(RIGHT_IP, 29999)
        left = DobotApiDashboard(LEFT_IP, 29999)
        port = PortHandler(os.environ.get("DOBOT_LEFT_GRIPPER_PORT", "COM5"))
        if not port.setBaudRate(1000000):
            raise RuntimeError("COM5 gripper unavailable")
        servo = sms_sts(port)
        motion = DobotApiMove(LEFT_IP, 30003)
        initial = joints(left)
        ensure_ready(left, log)
        pose = load_pose()
        if pose is None:
            print("No saved start pose yet; press S once and it will be recorded.", flush=True)
        else:
            print("Saved start pose: " + ", ".join(f"{v:.3f}" for v in pose), flush=True)
        print(f"{POLICY_NAME.upper()} loaded from {getattr(policy, 'run_dir', '?')}. "
              "Live cameras open. S=START  R=RETURN  Q=QUIT", flush=True)
        while True:
            im_left, im_top = frames_from(readers)
            key = show(im_left, im_top, "S=START  R=RETURN  Q=QUIT  clear the work area")
            if key in (ord("q"), 27):
                return
            if key == ord("r"):
                if pose is None:
                    continue
                if return_to_start(left, motion, pose, readers, log):
                    print("Returned to the saved start pose.", flush=True)
                continue
            if key == ord("s"):
                break
        if pose is None:
            # Recorded once: R keeps returning to the same reference pose.
            save_pose(joints(left))
            pose = load_pose()
            print("Recorded the current pose as the R reference.", flush=True)
        # A fresh observation and target are computed after S, not before.
        if mode(left) == 4:
            checked(left.EnableRobot(), "EnableRobot")
            enabled_here = True
        if mode(left) != 5:
            raise RuntimeError("Left arm failed to enter enabled idle mode")
        checked(left.SpeedFactor(SPEED_PERCENT), "SpeedFactor")
        checked(left.SpeedJ(SPEED_PERCENT), "SpeedJ")
        checked(left.AccJ(SPEED_PERCENT), "AccJ")
        # Start from a known gripper state: fully open, reported as 1.00.
        log["start_gripper_goal"] = grip_write(servo, GRIPPER_OPEN)
        time.sleep(0.2)
        log["start_gripper_readback"] = grip_read(servo)
        history = []
        step = 0
        prev_grip_target = None
        prev_target = None
        hold_until = 0.0
        chunk = None
        last_infer = 0.0
        infer_period = 1.0 / CONTROL_HZ
        min_infer_period = (2.0 if POLICY_NAME == "fm" else 1.0) / CONTROL_HZ
        last_tick = None
        servo_t = SERVO_T
        closed_seen = False
        release_time = None
        stop_reason = None
        started = time.monotonic()
        next_tick = started
        while True:
            im_left, im_top = frames_from(readers)
            arm = joints(left)
            gripper = grip_read(servo)
            if mode(left) not in (5, 7):
                raise RuntimeError("Left robot left running/idle mode")
            tick_now = time.monotonic()
            if last_tick is not None:
                # ServoJ interpolates over servo_t; if that is shorter than the actual
                # tick, the arm stops between commands and looks like it is stuttering.
                servo_t = min(0.1, max(0.03, 1.2 * (tick_now - last_tick)))
            last_tick = tick_now
            # Re-inference costs 60 ms (ACT) to a few hundred ms (flow matching), so
            # the control tick stays at 15 Hz and the policy is queried as often as it
            # can keep up. Older chunks keep contributing through the ensemble.
            if chunk is None or time.monotonic() - last_infer >= infer_period:
                qpos = np.concatenate((joints(right), [0.996], arm, [gripper])).astype(np.float32)
                t_infer = time.monotonic()
                if POLICY_NAME == "fm":
                    # Sampling from fresh noise every tick is what made the arm shake.
                    chunk = policy.predict(im_left, im_top, qpos,
                                           seed=FLOW_SEED if FLOW_SEED >= 0 else None)["left_arm_7"]
                    chunk = smooth_chunk(chunk, FM_SMOOTH_PASSES)
                else:
                    chunk = policy.predict(im_left, im_top, qpos)["left_arm_7"]
                last_infer = time.monotonic()
                infer_period = max(min_infer_period, (last_infer - t_infer) * 1.15)
                if not np.isfinite(chunk).all():
                    raise RuntimeError("Nonfinite policy action")
                log["inferences"] += 1
                log["infer_ms"] = round((last_infer - t_infer) * 1000, 1)
                history.append((step, chunk))
            while history and step - history[0][0] >= len(chunk):
                history.pop(0)
            total = np.zeros(7, dtype=np.float64)
            weight_sum = 0.0
            for start_step, predicted in history:
                age = step - start_step
                if 0 <= age < len(predicted):
                    weight = math.exp(-ENSEMBLE_DECAY * age)
                    total += weight * predicted[age]
                    weight_sum += weight
            if not weight_sum:
                raise RuntimeError("Empty ensemble")
            action = total / weight_sum
            target_arm = action[:6].astype(np.float64)
            target_grip = float(np.clip(action[6], 0, 1))
            if prev_target is not None:
                target_arm = EMA_ALPHA * target_arm + (1.0 - EMA_ALPHA) * prev_target
            prev_target = target_arm.copy()
            now = time.monotonic()
            if prev_grip_target is not None and abs(target_grip - prev_grip_target) > GRASP_HOLD_TRIGGER:
                hold_until = max(hold_until, now + GRASP_HOLD_S)
            prev_grip_target = target_grip
            holding = now < hold_until
            if holding:
                # Jaws still travelling: hold the arm so the grasp closes before the
                # plan lifts the tool away.
                target_arm = arm.copy()
                prev_target = arm.copy()
                log["hold_steps"] = log.get("hold_steps", 0) + 1
            entry = {"step": step, "time": time.time(), "elapsed_s": time.monotonic() - started,
                     "measured_arm": arm.tolist(), "measured_gripper": gripper,
                     "newest_chunk": chunk.tolist(), "ensemble_terms": len(history),
                     "oldest_age": step - history[0][0], "grasp_hold": holding,
                     "target_arm": target_arm.tolist(), "target_gripper": target_grip,
                     "planned_gap_rad": float(np.abs(target_arm - arm).max())}
            # A motion port timeout can mean the command was accepted. Never retry it.
            entry["arm_command_attempted"] = True
            entry["servo_t"] = round(servo_t, 3)
            reply = motion.ServoJ(*np.rad2deg(target_arm).tolist(), servo_t)
            entry["arm_reply"] = str(reply)
            if reply and not str(reply).startswith("0,"):
                raise RuntimeError(f"ServoJ rejected: {reply!r}")
            if abs(target_grip - gripper) >= 0.01:
                entry["gripper_command_attempted"] = True
                entry["gripper_goal"] = grip_write(servo, target_grip)
            # One grasp-release cycle is the task. Without this the policy, back at a
            # scene that looks like the start, starts the whole demonstration again.
            if gripper < 0.6:
                closed_seen = True
            if closed_seen and gripper > 0.85 and release_time is None:
                release_time = time.monotonic()
                entry["release_detected"] = True
            if AUTO_STOP_AFTER_RELEASE and release_time is not None \
                    and time.monotonic() - release_time > RELEASE_HOLD_S:
                stop_reason = "released after grasp"
                log["stop_reason"] = stop_reason
                entry["stop_reason"] = stop_reason
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            step += 1
            log["steps_completed"] = step
            elapsed = time.monotonic() - started
            heading = (f"step {step}  {elapsed:5.0f}s  terms {len(history):2d}  "
                       f"gap {entry['planned_gap_rad']:.2f}rad  grip {gripper:.2f}->{target_grip:.2f}  Q=STOP")
            key = show(im_left, im_top, heading)
            if key in (ord("q"), 27):
                break
            if stop_reason:
                print(f"Stopping: {stop_reason}.", flush=True)
                break
            next_tick += 1.0 / CONTROL_HZ
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
            if step % 60 == 0:
                print(f"{POLICY_NAME.upper()} step {step}, {elapsed:.0f}s, arm moved "
                      f"{float(np.abs(arm - initial).max()):.3f} rad. Q to stop.", flush=True)
        log["duration_s"] = time.monotonic() - started
        log["final_arm"] = joints(left).tolist()
        log["final_gripper"] = grip_read(servo)
        log["max_joint_excursion_rad"] = float(np.abs(np.array(log["final_arm"]) - initial).max())
        print(f"Rollout stopped after {step} steps ({log['duration_s']:.0f}s). "
              f"Summary: {summary_path}", flush=True)
    except BaseException as exc:
        log["error"] = repr(exc)
        raise
    finally:
        if enabled_here and left is not None:
            try:
                log["disable_reply"] = left.DisableRobot()
            except Exception as exc:
                log["disable_error"] = repr(exc)
        stream.close()
        try:
            summary_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        if motion is not None:
            motion.close()
        if left is not None:
            left.close()
        if right is not None:
            right.close()
        if port is not None and port.ser is not None:
            port.closePort()
        for reader in readers.values():
            reader.stop()
        for camera in cameras.values():
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
