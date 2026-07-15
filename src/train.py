import argparse
import csv
import os
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import get_datasets
from .evaluate import evaluate_all
from .models import get_model
from .poison import CCDTPoisonedDataset, make_poison_index_splits
from .triggers import get_default_configs, get_fixed_diverse_configs
from .config_search import search_dynamic_configs
from .utils import ensure_dir, get_device, load_yaml, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, default="full",
                        choices=["single", "naive_dual", "wo_invalid", "full"])
    parser.add_argument("--poison-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kc", type=int, default=None)
    parser.add_argument("--position-mode", type=str, default="fixed", choices=["fixed", "fixed_diverse", "dynamic"])
    parser.add_argument("--search-grid", type=int, default=4)
    parser.add_argument("--search-subset", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    epoch,
):
    model.train()

    total_loss = 0.0
    total_num = 0
    correct = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", ncols=100)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_num += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

        pbar.set_postfix({
            "loss": f"{total_loss / max(total_num, 1):.4f}",
            "acc": f"{correct / max(total_num, 1):.4f}",
        })

    return {
        "train_loss": total_loss / max(total_num, 1),
        "train_acc": correct / max(total_num, 1),
    }


def append_csv(path: str, row: Dict):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.kc is not None:
        cfg["kc"] = args.kc
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr

    set_seed(args.seed)
    device = get_device()

    dataset_name = cfg["dataset"]
    model_name = cfg["model"]
    kc = int(cfg["kc"])
    target_labels = [int(x) for x in cfg["target_labels"][:kc]]

    position_tag = {
        "fixed": "",
        "fixed_diverse": "_fdiv",
        "dynamic": "_dyn",
    }[args.position_mode]

    run_name = (
        f"{dataset_name}_{model_name}_"
        f"{args.method}{position_tag}_pr{args.poison_rate}_kc{kc}_seed{args.seed}"
    )

    output_dir = os.path.join(cfg["output_root"], run_name)
    ckpt_dir = os.path.join(cfg["checkpoint_root"], run_name)

    ensure_dir(output_dir)
    ensure_dir(ckpt_dir)

    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Method: {args.method}")
    print(f"Poison rate: {args.poison_rate}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    train_base, test_set, num_classes = get_datasets(
        dataset_name=dataset_name,
        data_root=cfg["data_root"],
        image_size=int(cfg["image_size"]),
    )

    if args.position_mode == "fixed":
        configs = get_default_configs(
            kc=kc,
            target_labels=target_labels,
        )
    elif args.position_mode == "fixed_diverse":
        configs = get_fixed_diverse_configs(
            kc=kc,
            target_labels=target_labels,
            image_size=int(cfg["image_size"]),
            trigger_size=int(cfg["trigger_size"]),
            margin=int(cfg["trigger_margin"]),
        )
    else:
        configs = search_dynamic_configs(
            dataset=train_base,
            kc=kc,
            target_labels=target_labels,
            image_size=int(cfg["image_size"]),
            trigger_size=int(cfg["trigger_size"]),
            margin=int(cfg["trigger_margin"]),
            grid_size=int(args.search_grid),
            subset_size=int(args.search_subset),
            seed=int(args.seed),
        )

    poison_splits = make_poison_index_splits(
        dataset=train_base,
        poison_rate=args.poison_rate,
        configs=configs,
        seed=args.seed,
    )

    train_set = CCDTPoisonedDataset(
        base_dataset=train_base,
        configs=configs,
        poison_splits=poison_splits,
        method=args.method,
        image_size=int(cfg["image_size"]),
        trigger_size=int(cfg["trigger_size"]),
        trigger_margin=int(cfg["trigger_margin"]),
    )

    poison_stats = {
        "clean_train_size": len(train_base),
        "expanded_train_size": len(train_set),
        "poison_record_size": len(train_set.records),
        "poison_splits": {str(k): len(v) for k, v in poison_splits.items()},
        "configs": configs,
    }

    save_json(poison_stats, os.path.join(output_dir, "poison_stats.json"))

    print("Poison stats:")
    print(poison_stats)

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )

    model = get_model(model_name, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["epochs"]),
    )

    scaler = GradScaler(enabled=device.type == "cuda")

    best_csg = -999.0
    best_metrics = None
    history_csv = os.path.join(output_dir, "history.csv")

    start_time = time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            epoch=epoch,
        )

        scheduler.step()

        do_eval = (
            epoch == 1
            or epoch % int(cfg["eval_every"]) == 0
            or epoch == int(cfg["epochs"])
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
        }

        if do_eval:
            eval_metrics = evaluate_all(
                model=model,
                loader=test_loader,
                configs=configs,
                target_labels=target_labels,
                device=device,
                image_size=int(cfg["image_size"]),
                trigger_size=int(cfg["trigger_size"]),
                trigger_margin=int(cfg["trigger_margin"]),
            )
            row.update(eval_metrics)

            print(
                f"[Eval epoch {epoch}] "
                f"CDA={eval_metrics['clean_acc']:.4f} "
                f"ValidASR={eval_metrics['valid_asr']:.4f} "
                f"WrongASR={eval_metrics['wrong_asr']:.4f} "
                f"SingleLeak={eval_metrics['single_leak']:.4f} "
                f"InvalidLeak={eval_metrics['invalid_leak']:.4f} "
                f"CSG={eval_metrics['csg']:.4f}"
            )

            if eval_metrics["csg"] > best_csg:
                best_csg = eval_metrics["csg"]
                best_metrics = eval_metrics
                if os.environ.get("SAVE_BEST", "0") == "1":
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "cfg": cfg,
                            "args": vars(args),
                            "configs": configs,
                            "metrics": eval_metrics,
                        },
                        os.path.join(ckpt_dir, "best.pt"),
                    )

        append_csv(history_csv, row)

    final_metrics = evaluate_all(
        model=model,
        loader=test_loader,
        configs=configs,
        target_labels=target_labels,
        device=device,
        image_size=int(cfg["image_size"]),
        trigger_size=int(cfg["trigger_size"]),
        trigger_margin=int(cfg["trigger_margin"]),
    )

    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "args": vars(args),
            "configs": configs,
            "metrics": final_metrics,
        },
        os.path.join(ckpt_dir, "last.pt"),
    )

    summary = {
        "run_name": run_name,
        "dataset": dataset_name,
        "model": model_name,
        "method": args.method,
        "poison_rate": args.poison_rate,
        "seed": args.seed,
        "kc": kc,
        "position_mode": args.position_mode,
        "search_grid": args.search_grid,
        "search_subset": args.search_subset,
        "target_labels": target_labels,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "elapsed_seconds": time.time() - start_time,
    }

    save_json(summary, os.path.join(output_dir, "summary.json"))

    print("=" * 80)
    print("Final metrics:")
    print(final_metrics)
    print("Best metrics:")
    print(best_metrics)
    print(f"Saved results to: {output_dir}")
    print(f"Saved checkpoints to: {ckpt_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
