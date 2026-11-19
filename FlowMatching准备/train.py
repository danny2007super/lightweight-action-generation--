"""Train the independent flow matching comparison on the ACT episode split."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import FlowPolicy, prepare_images

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ACT训练准备"))
from data import ACTDataset, load_plan  # noqa: E402


def evaluate(model, loader, device, image_size):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for image, state, action, padding in loader:
            image = prepare_images(image.to(device), image_size)
            state, action, padding = (x.to(device) for x in (state, action, padding))
            # Fixed noise/time for repeatable validation without moving train RNG.
            with torch.random.fork_rng(devices=[device.index or torch.cuda.current_device()]):
                torch.manual_seed(12345)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.flow_loss(image, state, action, padding)
            total += float(loss) * len(state)
            count += len(state)
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", type=Path, help="trusted last.pt from this trainer")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = Path(args.config).resolve().parent
    plan = load_plan(base / cfg["plan"])
    stats_path = base / cfg["act_stats"]
    if not stats_path.is_file():
        raise FileNotFoundError("ACT train-only stats are required for a matched comparison")
    stats = dict(np.load(stats_path))
    for key in ("camera_names", "state_dim", "action_dim"):
        if cfg[key] != plan[key]:
            raise ValueError(f"configuration disagrees with dataset: {key}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable; configure precision deliberately before training")
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.hub.set_dir(str(HERE.parents[1] / "work" / "torch-cache"))
    train_set = ACTDataset([plan["episodes"][i] for i in plan["train"]], cfg["camera_names"],
                           stats, cfg["chunk_size"], training=True)
    val_set = ACTDataset([plan["episodes"][i] for i in plan["validation"]], cfg["camera_names"],
                         stats, cfg["chunk_size"], training=False)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"])
    model = FlowPolicy(state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
                       horizon=cfg["chunk_size"], hidden=cfg["hidden_dim"],
                       layers=cfg["layers"], heads=cfg["heads"]).cuda()
    groups = [dict(params=model.vision.parameters(), lr=cfg["backbone_learning_rate"]),
              dict(params=(p for n, p in model.named_parameters() if not n.startswith("vision.")),
                   lr=cfg["learning_rate"])]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    first_update, best = 1, float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["config"] != cfg:
            raise ValueError("resume configuration differs from saved run")
        source_run = args.resume.resolve().parent
        old_split = json.loads((source_run / "dataset_split.json").read_text(encoding="utf-8"))
        if old_split != {"train": plan["train"], "validation": plan["validation"]}:
            raise ValueError("resume dataset split differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_update = int(checkpoint["update"]) + 1
        saved_best = torch.load(source_run / "best.pt", map_location="cpu", weights_only=True)
        best_update = int(saved_best["update"])
        del saved_best
        scores = []
        for metrics_path in source_run.parent.glob("run_fullres*/metrics.jsonl"):
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["update"] == best_update and "validation_flow_mse" in row:
                    scores.append(row["validation_flow_mse"])
        if not scores:
            raise ValueError("cannot find validation score for carried best.pt")
        best = min(scores)
        shutil.copy2(source_run / "best.pt", out / "best.pt")
        print(json.dumps({"resumed_after_update": first_update - 1, "prior_best_validation_flow_mse": best}), flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "dataset_split.json").write_text(json.dumps({"train": plan["train"],
                                                        "validation": plan["validation"]}), encoding="utf-8")
    iterator = iter(train_loader)
    model.train()
    for update in range(first_update, cfg["max_updates"] + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(cfg["gradient_accumulation"]):
            try:
                image, state, action, padding = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                image, state, action, padding = next(iterator)
            image = prepare_images(image.cuda(non_blocking=True), cfg["image_size"])
            state, action, padding = (x.cuda(non_blocking=True) for x in (state, action, padding))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.flow_loss(image, state, action, padding) / cfg["gradient_accumulation"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at update {update}")
            loss.backward()
            train_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = {"update": update, "train_flow_mse": train_loss}
        if update == 1 or update % cfg["validate_every"] == 0:
            score = evaluate(model, val_loader, torch.device("cuda"), cfg["image_size"])
            record["validation_flow_mse"] = score
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "config": cfg, "update": update}, out / "best.pt")
        if update % cfg["save_every"] == 0 or update == cfg["max_updates"]:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": cfg, "update": update}, out / "last.pt")
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if update == 1 or update % 10 == 0:
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
