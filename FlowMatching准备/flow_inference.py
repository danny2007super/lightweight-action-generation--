"""Observation-to-action flow matching interface; does not command a robot."""
from pathlib import Path
import json

import numpy as np
import torch

from model import FlowPolicy, prepare_images

HERE = Path(__file__).resolve().parent


class FlowInference:
    def __init__(self, run_dir=HERE / "runs" / "run_fullres_resume_002", checkpoint="best.pt", device="cuda"):
        self.run_dir = Path(run_dir).resolve()
        self.config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        if self.config["camera_names"] != ["left", "top"]:
            raise ValueError("camera order mismatch")
        if (self.config["state_dim"], self.config["action_dim"]) != (14, 14):
            raise ValueError("state/action dimension mismatch")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        with np.load(HERE.parent / "ACT训练准备" / "runs" / "run_001" / "dataset_stats.npz") as data:
            self.stats = {key: data[key] for key in data.files}
        cfg = self.config
        self.model = FlowPolicy(state_dim=14, action_dim=14, horizon=cfg["chunk_size"],
                                hidden=cfg["hidden_dim"], layers=cfg["layers"],
                                heads=cfg["heads"], pretrained=False).to(self.device)
        saved = torch.load(self.run_dir / checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(saved["model"])
        self.checkpoint_update = int(saved["update"])
        self.model.eval()

    def predict(self, left_rgb, top_rgb, qpos, *, seed=None):
        """Input raw RGB uint8 (480,640,3) x2 and qpos[14]; return proposed chunks."""
        images = []
        for name, image in (("left_rgb", left_rgb), ("top_rgb", top_rgb)):
            image = np.asarray(image)
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f"{name} must be RGB uint8 [480,640,3]")
            images.append(image)
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError("qpos must contain 14 finite values")
        image = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).to(self.device).float()[None] / 255
        image = prepare_images(image, self.config["image_size"])
        state = torch.from_numpy((qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]).to(self.device)[None]
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None), torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                normalized = self.model.sample(image, state, self.config["integration_steps"])[0].float().cpu().numpy()
        actions = normalized * self.stats["action_std"] + self.stats["action_mean"]
        if actions.shape != (self.config["chunk_size"], 14) or not np.isfinite(actions).all():
            raise ValueError("invalid model output")
        return {"actions_14": actions, "left_arm_7": actions[:, 7:14],
                "checkpoint_update": self.checkpoint_update,
                "camera_order": ["left", "top"], "nominal_hz": 15}
