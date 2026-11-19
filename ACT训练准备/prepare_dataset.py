"""Inspect new files and write an external plan. Never modifies source HDF5."""
import argparse
import json
from pathlib import Path
import numpy as np
from data import inspect_file


def prepare(directory, output, cameras, validation_ratio=.2, seed=0, accepted_files=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"use a new plan filename: {output}")
    paths = sorted(set(directory.glob("*.hdf5")) | set(directory.glob("*.h5")))
    accepted = set(accepted_files) if accepted_files is not None else None
    if accepted is not None:
        missing = accepted - {p.name for p in paths}
        if missing:
            raise ValueError(f"accepted files not found: {sorted(missing)}")
    episodes, rejected = [], []
    for path in paths:
        if accepted is not None and path.name not in accepted:
            continue
        try:
            entry = inspect_file(path, cameras)
            if entry["status"] == "failure":
                rejected.append(dict(file=path.name, reason="source marked failure"))
                continue
            episodes.append(entry)
        except Exception as exc:
            rejected.append(dict(file=path.name, reason=str(exc)))
    if len(episodes) < 2:
        raise ValueError(f"need at least two eligible episodes; rejected={rejected}")
    dims = {(e["state_dim"], e["action_dim"], tuple(e["camera_names"]), tuple(e["image_shape"])) for e in episodes}
    semantics = {json.dumps({key:e["metadata"].get(key) for key in (
        "task", "channel_order", "joint_unit", "gripper", "config_sha256", "camera_serials",
        "camera_flips", "image_pipeline_version", "task_protocol_version")},sort_keys=True) for e in episodes}
    if len(dims) != 1 or len(semantics) != 1:
        raise ValueError("incompatible dimensions, camera geometry or collection configuration; separate dataset versions")
    if len({e["source_episode"] for e in episodes}) != len(episodes):
        raise ValueError("duplicate source episodes; deduplicate before splitting")
    if len({e["numeric_sha256"] for e in episodes}) != len(episodes):
        raise ValueError("duplicate state/action trajectories; deduplicate before splitting")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    ids = np.random.default_rng(seed).permutation(len(episodes)).tolist()
    nval = min(len(ids)-1, max(1, round(len(ids)*validation_ratio)))
    first = episodes[0]
    plan = dict(schema_version=1, directory=str(directory), episodes=episodes, rejected=rejected,
                camera_names=first["camera_names"], state_dim=first["state_dim"], action_dim=first["action_dim"],
                train=sorted(ids[nval:]), validation=sorted(ids[:nval]), seed=seed,
                action_alignment="same_row_no_shift", normalization="train_episodes_only",
                unreviewed_count=sum(e["status"] in ("unknown", "unreviewed") for e in episodes))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    return plan


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output",required=True)
    parser.add_argument("--cameras",nargs="+",default=["left","top"])
    parser.add_argument("--validation-ratio",type=float,default=.2)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--accepted-list",help="optional JSON list of reviewed episode filenames")
    args=parser.parse_args()
    accepted=json.loads(Path(args.accepted_list).read_text(encoding="utf-8-sig")) if args.accepted_list else None
    plan=prepare(args.directory,args.output,args.cameras,args.validation_ratio,args.seed,accepted)
    print(json.dumps({"train":len(plan["train"]),"validation":len(plan["validation"]),
                      "rejected":plan["rejected"],"unreviewed":plan["unreviewed_count"],
                      "camera_names":plan["camera_names"],"dimensions":[plan["state_dim"],plan["action_dim"]]},ensure_ascii=False,indent=2))
