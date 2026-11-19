"""Read one recorded HDF5 observation and export ACT's proposed action chunk.

This script never connects to or commands a robot.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "ACT训练准备" / "runs" / "run_001"
sys.path.insert(0, str(HERE.parent / "ACT训练准备"))
from model import PreparedACTPolicy  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "prediction.json")
    args = parser.parse_args()
    cfg = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    with np.load(RUN / "dataset_stats.npz") as loaded:
        stats = {name: loaded[name] for name in loaded.files}
    with h5py.File(args.episode, "r") as f:
        count = len(f["observations/qpos"])
        if not 0 <= args.frame < count:
            raise ValueError(f"frame must be in 0..{count - 1}")
        if f.attrs.get("action_alignment") != "pre_command_observation_same_row_action":
            raise ValueError("unexpected observation/action alignment")
        qpos = f["observations/qpos"][args.frame].astype(np.float32)
        images = np.stack([f[f"observations/images/{name}"][args.frame]
                           for name in cfg["camera_names"]])
        recorded_action = f["action"][args.frame].astype(np.float32)
    if qpos.shape != (14,) or images.shape != (2, 480, 640, 3):
        raise ValueError("unexpected recorded state or image shape")
    if images.dtype != np.uint8 or not np.isfinite(qpos).all():
        raise ValueError("invalid input")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    # All weights come from the saved checkpoint; no pretrained download needed.
    cfg["pretrained_backbone"] = False
    policy = PreparedACTPolicy(cfg, device)
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint["policy"])
    chosen_update = int(checkpoint["update"])
    del checkpoint
    policy.eval()
    state = torch.from_numpy((qpos - stats["qpos_mean"]) / stats["qpos_std"]).unsqueeze(0).to(device)
    image = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        normalized = policy(state, image)[0].float().cpu().numpy()
    actions = normalized * stats["action_std"] + stats["action_mean"]
    if actions.shape != (30, 14) or not np.isfinite(actions).all():
        raise ValueError("invalid policy output")
    result = {
        "source": str(args.episode.resolve()), "frame": args.frame,
        "checkpoint_update": chosen_update, "camera_order": cfg["camera_names"],
        "action_alignment": "pre_command_observation_same_row_action",
        "input_qpos_14": qpos.tolist(),
        "recorded_action_at_frame_14": recorded_action.tolist(),
        "predicted_actions_14": actions.tolist(),
        "predicted_active_left_arm_7": actions[:, 7:14].tolist(),
        "note": "Prediction only. No robot command or joint-limit validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(actions)} proposed actions to {args.output}")


if __name__ == "__main__":
    main()
