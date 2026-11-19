"""Dobot ACT HDF5 loader: explicit cameras, same-row targets, train-only stats."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

ALIGNMENT = "pre_command_observation_same_row_action"


def inspect_file(path, cameras=None):
    path = Path(path).resolve()
    with h5py.File(path, "r") as f:
        if bool(f.attrs.get("compress", False)):
            raise ValueError(f"{path.name}: JPEG-packed schema needs a decoder adapter")
        alignment = f.attrs.get("action_alignment")
        if alignment != ALIGNMENT:
            raise ValueError(f"{path.name}: action_alignment is {alignment!r}; confirm alignment before conversion")
        q, a = f["observations/qpos"][()], f["action"][()]
        if q.ndim != 2 or a.ndim != 2 or len(q) != len(a) or len(a) < 2:
            raise ValueError(f"{path.name}: inconsistent qpos/action lengths")
        if q.shape[1] not in (7, 14) or q.shape[1] != a.shape[1]:
            raise ValueError(f"{path.name}: expected equal 7- or 14-dimensional state and action")
        if not np.isfinite(q).all() or not np.isfinite(a).all():
            raise ValueError(f"{path.name}: non-finite state or action")
        for values in (q, a):
            gripper = values[:, [6]] if values.shape[1] == 7 else values[:, [6, 13]]
            if np.any((gripper < -1e-5) | (gripper > 1 + 1e-5)):
                raise ValueError(f"{path.name}: gripper must use [0,1]; confirm source semantics")
        names = list(cameras) if cameras else sorted(f["observations/images"].keys())
        if not names or len(set(names)) != len(names):
            raise ValueError("camera names must be nonempty and unique")
        geometries = []
        for name in names:
            ds = f[f"observations/images/{name}"]
            if len(ds.shape) != 4 or ds.shape[0] != len(a) or ds.shape[-1] != 3 or ds.dtype != np.dtype("uint8"):
                raise ValueError(f"{path.name}: invalid image schema for {name}: {ds.shape} {ds.dtype}")
            _ = ds[0], ds[-1]
            geometries.append(list(ds.shape[1:]))
        if len({tuple(g) for g in geometries}) != 1:
            raise ValueError(f"{path.name}: cameras have different image sizes")
        timestamp = f["timestamp"][()] if "timestamp" in f else None
        if timestamp is not None and (timestamp.shape != (len(a),) or not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0)):
            raise ValueError(f"{path.name}: invalid or non-increasing timestamps")
        metadata = json.loads(f.attrs.get("metadata_json", "{}"))
        stat = path.stat()
        return dict(path=str(path), frames=len(a), state_dim=q.shape[1], action_dim=a.shape[1],
                    camera_names=names, image_shape=geometries[0],
                    status=metadata.get("status", "unknown"), task=metadata.get("task"),
                    source_episode=str(f.attrs.get("source_episode", path.stem)),
                    metadata=metadata, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    numeric_sha256=hashlib.sha256(q.tobytes() + a.tobytes()).hexdigest(),
                    duration_s=float(timestamp[-1]-timestamp[0]) if timestamp is not None else None)


def training_stats(episodes):
    stats = {}
    for key, name in (("observations/qpos", "qpos"), ("action", "action")):
        count, total, squares = 0, None, None
        for episode in episodes:
            with h5py.File(episode["path"], "r") as f:
                values = f[key][()].astype(np.float64)
            if total is None:
                total, squares = np.zeros(values.shape[1]), np.zeros(values.shape[1])
            total += values.sum(0)
            squares += (values * values).sum(0)
            count += len(values)
        if not count:
            raise ValueError("empty training split")
        mean = total / count
        stats[name + "_mean"] = mean.astype(np.float32)
        stats[name + "_std"] = np.sqrt(np.maximum(squares/count - mean**2, .0001)).astype(np.float32)
    return stats


def load_window(path, cameras, start, chunk_size, stats):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    with h5py.File(path, "r") as f:
        n = len(f["action"])
        if not 0 <= start < n:
            raise ValueError("invalid window start")
        image = np.stack([f[f"observations/images/{camera}"][start] for camera in cameras])
        state = f["observations/qpos"][start].astype(np.float32)
        count = min(chunk_size, n - start)
        targets = np.zeros((chunk_size, f["action"].shape[1]), np.float32)
        targets[:count] = (f["action"][start:start+count] - stats["action_mean"]) / stats["action_std"]
    state = (state - stats["qpos_mean"]) / stats["qpos_std"]
    mask = np.arange(chunk_size) >= count
    return (torch.from_numpy(image.transpose(0, 3, 1, 2).copy()).float()/255.,
            torch.from_numpy(state), torch.from_numpy(targets), torch.from_numpy(mask))


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not plan["train"] or not plan["validation"]:
        raise ValueError("train and validation splits must both contain whole episodes")
    if set(plan["train"]) & set(plan["validation"]):
        raise ValueError("overlapping train/validation episodes")
    selected = [plan["episodes"][i] for i in plan["train"]+plan["validation"]]
    source_groups = {}
    for split in ("train", "validation"):
        for i in plan[split]:
            entry = plan["episodes"][i]
            for key in ("source_episode", "numeric_sha256"):
                group = (key, entry[key])
                if group in source_groups and source_groups[group] != split:
                    raise ValueError(f"source/duplicate leakage between splits: {entry['path']}")
                source_groups[group] = split
    for entry in selected:
        stat = Path(entry["path"]).stat()
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"dataset changed after preparation: {entry['path']}")
    return plan


class ACTDataset(Dataset):
    def __init__(self, episodes, cameras, stats, chunk_size, *, training, windows_per_episode=8):
        self.episodes, self.cameras, self.stats = episodes, list(cameras), stats
        self.chunk_size, self.training = chunk_size, training
        self.windows = windows_per_episode if training else 3
        if self.windows < 1:
            raise ValueError("windows_per_episode must be positive")

    def __len__(self):
        return len(self.episodes) * self.windows

    def __getitem__(self, index):
        episode = self.episodes[index // self.windows]
        if self.training:
            start = int(np.random.randint(episode["frames"]))
        else:
            start = int(round((episode["frames"]-1) * (index % self.windows) / max(self.windows-1, 1)))
        return load_window(episode["path"], self.cameras, start, self.chunk_size, self.stats)
