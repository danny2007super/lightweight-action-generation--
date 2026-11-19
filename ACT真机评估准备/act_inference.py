"""Reusable ACT observation-to-action interface; no robot transport or commands."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "ACT训练准备"
DEFAULT_RUN = PROJECT / "runs" / "run_001"
sys.path.insert(0, str(PROJECT))
from model import PreparedACTPolicy  # noqa: E402


class ACTInference:
    """Load one checkpoint once, then predict from live-compatible observations."""

    def __init__(self, run_dir=DEFAULT_RUN, checkpoint="best.pt", device="cuda"):
        self.run_dir = Path(run_dir).resolve()
        self.config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        with np.load(self.run_dir / "dataset_stats.npz") as loaded:
            self.stats = {key: loaded[key] for key in loaded.files}
        if self.config["camera_names"] != ["left", "top"]:
            raise ValueError("expected left, top camera order")
        if (self.config["state_dim"], self.config["action_dim"]) != (14, 14):
            raise ValueError("expected 14-dimensional state and action")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.device = torch.device(device)
        config = dict(self.config)
        config["pretrained_backbone"] = False  # checkpoint provides every model weight
        self.policy = PreparedACTPolicy(config, self.device)
        saved = torch.load(self.run_dir / checkpoint, map_location="cpu", weights_only=True)
        self.policy.load_state_dict(saved["policy"])
        self.checkpoint_update = int(saved["update"])
        del saved
        self.policy.eval()

    def predict(self, left_rgb, top_rgb, qpos):
        """Return proposed actions in demonstration units, without executing them.

        Inputs: two RGB uint8 arrays [480,640,3], qpos float[14] ordered as
        recorded (inactive arm first 7, active left arm last 7).
        Returns: dict with numpy arrays `actions_14` [30,14] and
        `left_arm_7` [30,7]. Caller must validate limits and control mode.
        """
        frames = []
        for name, value in (("left_rgb", left_rgb), ("top_rgb", top_rgb)):
            frame = np.asarray(value)
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise ValueError(f"{name} must be RGB uint8 [480,640,3]")
            frames.append(frame)
        state = np.asarray(qpos, dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("qpos must be 14 finite values in demonstration order")
        image = np.stack(frames).transpose(0, 3, 1, 2).copy()
        image = torch.from_numpy(image).unsqueeze(0).to(self.device).float() / 255.0
        normalized_state = (state - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        normalized_state = torch.from_numpy(normalized_state).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast(self.device.type,
                                                     dtype=torch.bfloat16,
                                                     enabled=self.device.type == "cuda"):
            normalized_action = self.policy(normalized_state, image)[0].float().cpu().numpy()
        actions = normalized_action * self.stats["action_std"] + self.stats["action_mean"]
        if actions.shape != (self.config["chunk_size"], 14) or not np.isfinite(actions).all():
            raise ValueError("invalid model output")
        return {"actions_14": actions, "left_arm_7": actions[:, 7:14],
                "checkpoint_update": self.checkpoint_update,
                "nominal_hz": 15, "camera_order": ["left", "top"]}
