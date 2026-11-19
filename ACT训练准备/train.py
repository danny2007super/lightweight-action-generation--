"""Train upstream ACT offline against a prepared HDF5 plan."""
import argparse
import contextlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from data import ACTDataset, load_plan, training_stats
from model import PreparedACTPolicy, configure_cache


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate(policy, loader, device, use_amp):
    policy.eval()
    totals, count = {}, 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Fixed validation windows and fixed CVAE sampling do not consume training RNG.
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(12345)
        for batch in loader:
            image, state, targets, mask = [x.to(device) for x in batch]
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,enabled=use_amp):
                loss = policy(state,image,targets,mask)
            for key,value in loss.items():
                totals[key]=totals.get(key,0.)+float(value)*len(state)
            count+=len(state)
    policy.train()
    return {key:value/count for key,value in totals.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--plan",help="plan generated with prepare_dataset.py")
    parser.add_argument("--output",required=True,help="new output directory")
    parser.add_argument("--resume",help="trusted checkpoint produced by this trainer")
    parser.add_argument("--max-updates",type=int)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    plan_path=args.plan or config.get("dataset_plan")
    if not plan_path:
        parser.error("provide --plan after receiving and checking the new HDF5 dataset")
    plan=load_plan(plan_path)
    for key in ("camera_names","state_dim","action_dim"):
        config[key]=plan[key]
    if args.max_updates is not None:
        config["max_updates"]=args.max_updates
    for key in ("chunk_size","batch_size","gradient_accumulation","max_updates","validate_every","save_every"):
        if config[key]<1:
            raise ValueError(f"{key} must be positive")
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("use a new output directory, including when resuming")
    output.mkdir(parents=True)
    device=torch.device(config["device"])
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run smoke_test.py to diagnose the environment")
    use_amp=bool(config["mixed_precision"] and device.type=="cuda")
    seed_all(config["seed"])
    configure_cache(Path(__file__).parents[2]/"work"/"torch-cache")
    stats=training_stats([plan["episodes"][i] for i in plan["train"]])
    np.savez(output/"dataset_stats.npz",**stats)
    (output/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"dataset_plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    train=ACTDataset([plan["episodes"][i] for i in plan["train"]],plan["camera_names"],stats,config["chunk_size"],
                     training=True,windows_per_episode=config["windows_per_episode"])
    val=ACTDataset([plan["episodes"][i] for i in plan["validation"]],plan["camera_names"],stats,config["chunk_size"],training=False)
    train_loader=DataLoader(train,batch_size=config["batch_size"],shuffle=True,num_workers=config["num_workers"],pin_memory=device.type=="cuda")
    val_loader=DataLoader(val,batch_size=config["batch_size"],num_workers=config["num_workers"])
    policy=PreparedACTPolicy(config,device)
    optimizer=policy.configure_optimizers()
    amp_dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=use_amp and amp_dtype==torch.float16)
    first_update,best=1,float("inf")
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device,weights_only=True)
        old=checkpoint["config"]
        for key in ("camera_names","state_dim","action_dim","chunk_size","hidden_dim","dim_feedforward","encoder_layers","decoder_layers"):
            if old[key]!=config[key]:
                raise ValueError(f"resume configuration differs: {key}")
        if checkpoint.get("split_sources") != {split:[plan["episodes"][i]["numeric_sha256"] for i in plan[split]] for split in ("train","validation")}:
            raise ValueError("resume dataset or train/validation split differs")
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        first_update,best=checkpoint["update"]+1,checkpoint["best_validation_loss"]
    iterator=iter(train_loader)
    log=output/"metrics.jsonl"
    last_update=first_update-1
    def payload(update):
        return dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    update=update,best_validation_loss=best,config=config,
                    split_sources={split:[plan["episodes"][i]["numeric_sha256"] for i in plan[split]] for split in ("train","validation")})
    policy.train()
    start=time.monotonic()
    try:
        for update in range(first_update,config["max_updates"]+1):
            optimizer.zero_grad(set_to_none=True)
            train_loss=0.
            for _ in range(config["gradient_accumulation"]):
                try:
                    batch=next(iterator)
                except StopIteration:
                    iterator=iter(train_loader)
                    batch=next(iterator)
                image,state,targets,mask=[x.to(device,non_blocking=True) for x in batch]
                with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
                    losses=policy(state,image,targets,mask)
                    loss=losses["loss"]/config["gradient_accumulation"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at update {update}")
                scaler.scale(loss).backward()
                train_loss+=float(loss.detach())
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),config["gradient_clip_norm"],error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            last_update=update
            metrics=dict(update=update,train_loss=train_loss,gradient_norm=float(norm),elapsed_s=time.monotonic()-start)
            if update==1 or update%config["validate_every"]==0:
                metrics["validation"]=evaluate(policy,val_loader,device,use_amp)
                score=metrics["validation"]["loss"]
                if not np.isfinite(score):
                    raise FloatingPointError("non-finite validation loss")
                if score<best:
                    best=score
                    atomic_checkpoint(output/"best.pt",payload(update))
            if update%config["save_every"]==0 or update==config["max_updates"]:
                atomic_checkpoint(output/"last.pt",payload(update))
            with log.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(metrics)+"\n")
            if update==1 or update%10==0 or "validation" in metrics:
                print(json.dumps(metrics),flush=True)
    except KeyboardInterrupt:
        atomic_checkpoint(output/"last.pt",payload(last_update))
        print(f"Stopped; checkpoint saved after completed update {last_update}",flush=True)
        return
    print(f"Finished: {output}",flush=True)


if __name__=="__main__":
    main()
