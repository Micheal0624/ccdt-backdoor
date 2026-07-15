from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adaptive_loss import FeedbackAdaptiveController, GROUP_NAMES, ID_TO_GROUP
from .config_search import search_dynamic_configs
from .datasets import get_datasets, get_feedback_dataset
from .evaluate import evaluate_all
from .models import get_model
from .poison import CCDTPoisonedDataset, make_poison_index_splits
from .triggers import get_default_configs, get_fixed_diverse_configs
from .utils import ensure_dir, get_device, load_yaml, save_json, set_seed


CORE_METRICS = (
    "clean_acc",
    "valid_asr",
    "wrong_asr",
    "single_leak",
    "invalid_leak",
    "csg",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CCDT with feedback-adaptive four-component loss balancing."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--poison-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kc", type=int, default=None)
    parser.add_argument(
        "--position-mode",
        type=str,
        default="dynamic",
        choices=["fixed", "fixed_diverse", "dynamic"],
    )
    parser.add_argument("--search-grid", type=int, default=4)
    parser.add_argument("--search-subset", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--feedback-size", type=int, default=2048)
    parser.add_argument(
        "--adaptive-mode",
        type=str,
        default="feedback",
        choices=["equal", "manual", "feedback"],
    )
    parser.add_argument("--feedback-ema", type=float, default=0.9)
    parser.add_argument("--feedback-temperature", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-weight", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=2.50)
    parser.add_argument("--target-valid-asr", type=float, default=0.99)
    parser.add_argument("--target-single-leak", type=float, default=0.02)
    parser.add_argument("--target-invalid-leak", type=float, default=0.02)
    parser.add_argument("--target-wrong-asr", type=float, default=0.01)
    parser.add_argument("--clean-tolerance", type=float, default=0.01)
    parser.add_argument("--asr-tolerance", type=float, default=0.01)
    parser.add_argument("--leak-tolerance", type=float, default=0.02)
    parser.add_argument("--wrong-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--manual-weights",
        type=str,
        default="1.0,1.5,0.75,1.25",
        help="clean,valid,single,invalid",
    )
    parser.add_argument("--run-tag", type=str, default="")
    return parser.parse_args()


def parse_manual_weights(text: str) -> Dict[str, float]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 4:
        raise ValueError("--manual-weights requires four comma-separated values")
    return dict(zip(GROUP_NAMES, values))


def atomic_torch_save(payload: Mapping[str, object], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_history_csv(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    ordered_keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temporary, path)


def prefixed_metrics(prefix: str, metrics: Mapping[str, object]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key in CORE_METRICS:
        if key in metrics:
            result[f"{prefix}_{key}"] = metrics[key]
    for index in range(1, 11):
        key = f"valid_asr_c{index}"
        if key in metrics:
            result[f"{prefix}_{key}"] = metrics[key]
    return result


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    epoch: int,
    weights: Mapping[str, float],
) -> Dict[str, float]:
    model.train()

    total_weighted_loss = 0.0
    total_num = 0
    correct = 0
    group_loss_sums = {name: 0.0 for name in GROUP_NAMES}
    group_counts = {name: 0 for name in GROUP_NAMES}

    pbar = tqdm(loader, desc=f"Adaptive train epoch {epoch}", ncols=120)

    for x, y, group_id in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        group_id = group_id.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(x)
            per_sample_loss = F.cross_entropy(logits, y, reduction="none")
            batch_size = max(int(y.numel()), 1)
            loss = per_sample_loss.new_zeros(())

            for numeric_id, group_name in ID_TO_GROUP.items():
                mask = group_id == int(numeric_id)
                if mask.any():
                    # Dividing by the total batch size is deliberate: when all
                    # lambdas equal one, this sum is exactly the original mean CE.
                    contribution = per_sample_loss[mask].sum() / batch_size
                    loss = loss + float(weights[group_name]) * contribution

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = int(y.numel())
        total_weighted_loss += float(loss.detach().item()) * bs
        total_num += bs
        correct += int((logits.argmax(dim=1) == y).sum().item())

        detached = per_sample_loss.detach()
        for numeric_id, group_name in ID_TO_GROUP.items():
            mask = group_id == int(numeric_id)
            count = int(mask.sum().item())
            if count:
                group_loss_sums[group_name] += float(detached[mask].sum().item())
                group_counts[group_name] += count

        pbar.set_postfix({
            "loss": f"{total_weighted_loss / max(total_num, 1):.4f}",
            "acc": f"{correct / max(total_num, 1):.4f}",
            "lv": f"{weights['valid']:.2f}",
            "ls": f"{weights['single']:.2f}",
            "li": f"{weights['invalid']:.2f}",
        })

    result: Dict[str, float] = {
        "train_loss": total_weighted_loss / max(total_num, 1),
        "train_acc": correct / max(total_num, 1),
    }

    for group_name in GROUP_NAMES:
        count = group_counts[group_name]
        result[f"train_count_{group_name}"] = float(count)
        result[f"train_loss_{group_name}"] = (
            group_loss_sums[group_name] / count if count > 0 else float("nan")
        )
        result[f"train_contribution_{group_name}"] = (
            group_loss_sums[group_name] / max(total_num, 1)
        )

    return result


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
    if args.eval_every is not None:
        cfg["eval_every"] = args.eval_every

    set_seed(args.seed)
    device = get_device()

    dataset_name = cfg["dataset"]
    model_name = cfg["model"]
    kc = int(cfg["kc"])
    target_labels = [int(value) for value in cfg["target_labels"][:kc]]

    position_tag = {
        "fixed": "fixed",
        "fixed_diverse": "fdiv",
        "dynamic": "dyn",
    }[args.position_mode]
    extra_tag = f"_{args.run_tag.strip()}" if args.run_tag.strip() else ""
    run_name = (
        f"{dataset_name}_{model_name}_adaptive_{args.adaptive_mode}_{position_tag}"
        f"_pr{args.poison_rate}_kc{kc}_seed{args.seed}{extra_tag}"
    )

    output_dir = os.path.join(cfg["output_root"], run_name)
    ckpt_dir = os.path.join(cfg["checkpoint_root"], run_name)
    ensure_dir(output_dir)
    ensure_dir(ckpt_dir)

    print("=" * 100)
    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Adaptive mode: {args.adaptive_mode}")
    print(f"Poison rate: {args.poison_rate}")
    print(f"Kc: {kc}")
    print(f"Seed: {args.seed}")
    print("=" * 100)

    train_base, test_set, num_classes = get_datasets(
        dataset_name=dataset_name,
        data_root=cfg["data_root"],
        image_size=int(cfg["image_size"]),
    )
    feedback_set = get_feedback_dataset(
        dataset_name=dataset_name,
        data_root=cfg["data_root"],
        image_size=int(cfg["image_size"]),
        subset_size=int(args.feedback_size),
        seed=int(args.seed),
    )

    if args.position_mode == "fixed":
        configs = get_default_configs(kc=kc, target_labels=target_labels)
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
        method="full",
        image_size=int(cfg["image_size"]),
        trigger_size=int(cfg["trigger_size"]),
        trigger_margin=int(cfg["trigger_margin"]),
        return_group=True,
    )

    poison_stats = {
        "clean_train_size": len(train_base),
        "expanded_train_size": len(train_set),
        "poison_record_size": len(train_set.records),
        "feedback_size": len(feedback_set),
        "feedback_source": "deterministic_subset_of_training_split",
        "poison_splits": {str(key): len(value) for key, value in poison_splits.items()},
        "configs": configs,
    }
    save_json(poison_stats, os.path.join(output_dir, "poison_stats.json"))

    loader_kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "pin_memory": True,
        "drop_last": False,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    feedback_loader = DataLoader(feedback_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = get_model(model_name, num_classes).to(device)
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

    controller = FeedbackAdaptiveController(
        mode=args.adaptive_mode,
        ema=float(args.feedback_ema),
        temperature=float(args.feedback_temperature),
        min_weight=float(args.min_weight),
        max_weight=float(args.max_weight),
        warmup_epochs=int(args.warmup_epochs),
        target_valid_asr=float(args.target_valid_asr),
        target_single_leak=float(args.target_single_leak),
        target_invalid_leak=float(args.target_invalid_leak),
        target_wrong_asr=float(args.target_wrong_asr),
        clean_tolerance=float(args.clean_tolerance),
        asr_tolerance=float(args.asr_tolerance),
        leak_tolerance=float(args.leak_tolerance),
        wrong_tolerance=float(args.wrong_tolerance),
        manual_weights=parse_manual_weights(args.manual_weights),
    )

    history_path = os.path.join(output_dir, "history.csv")
    controller_history_path = os.path.join(output_dir, "controller_history.json")
    history_rows: List[Dict[str, object]] = []
    controller_history: List[Dict[str, object]] = []
    best_feedback_csg = -float("inf")
    best_feedback_metrics = None
    best_test_metrics = None
    start_time = time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        weights_used = controller.current_weights()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            weights=weights_used,
        )
        scheduler.step()

        do_eval = (
            epoch == 1
            or epoch % int(cfg["eval_every"]) == 0
            or epoch == int(cfg["epochs"])
        )

        row: Dict[str, object] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
        }
        for name in GROUP_NAMES:
            row[f"lambda_{name}_used"] = weights_used[name]

        if do_eval:
            feedback_metrics = evaluate_all(
                model=model,
                loader=feedback_loader,
                configs=configs,
                target_labels=target_labels,
                device=device,
                image_size=int(cfg["image_size"]),
                trigger_size=int(cfg["trigger_size"]),
                trigger_margin=int(cfg["trigger_margin"]),
            )
            test_metrics = evaluate_all(
                model=model,
                loader=test_loader,
                configs=configs,
                target_labels=target_labels,
                device=device,
                image_size=int(cfg["image_size"]),
                trigger_size=int(cfg["trigger_size"]),
                trigger_margin=int(cfg["trigger_margin"]),
            )
            report = controller.update(epoch, feedback_metrics)
            controller_history.append(report)
            save_json({"updates": controller_history}, controller_history_path)

            row.update(prefixed_metrics("feedback", feedback_metrics))
            row.update(prefixed_metrics("test", test_metrics))
            row["controller_status"] = report["status"]
            for name in GROUP_NAMES:
                row[f"controller_score_{name}"] = report.get("scores", {}).get(name)

            if feedback_metrics["csg"] > best_feedback_csg:
                best_feedback_csg = float(feedback_metrics["csg"])
                best_feedback_metrics = dict(feedback_metrics)
                best_test_metrics = dict(test_metrics)
                if os.environ.get("SAVE_BEST", "0") == "1":
                    atomic_torch_save(
                        {
                            "model": model.state_dict(),
                            "cfg": cfg,
                            "args": vars(args),
                            "configs": configs,
                            "controller": controller.state_dict(),
                            "feedback_metrics": feedback_metrics,
                            "test_metrics": test_metrics,
                        },
                        os.path.join(ckpt_dir, "best.pt"),
                    )

            print(
                f"[Eval epoch {epoch}] "
                f"Feedback CSG={feedback_metrics['csg']:.4f} "
                f"Test ACC={test_metrics['clean_acc']:.4f} "
                f"ASR={test_metrics['valid_asr']:.4f} "
                f"Wrong={test_metrics['wrong_asr']:.4f} "
                f"Single={test_metrics['single_leak']:.4f} "
                f"Invalid={test_metrics['invalid_leak']:.4f} "
                f"CSG={test_metrics['csg']:.4f}"
            )
            print(
                "[Weights next] "
                + " ".join(
                    f"{name}={controller.weights[name]:.3f}" for name in GROUP_NAMES
                )
            )

        next_weights = controller.current_weights()
        for name in GROUP_NAMES:
            row[f"lambda_{name}_next"] = next_weights[name]

        history_rows.append(row)
        write_history_csv(history_path, history_rows)

    final_feedback_metrics = evaluate_all(
        model=model,
        loader=feedback_loader,
        configs=configs,
        target_labels=target_labels,
        device=device,
        image_size=int(cfg["image_size"]),
        trigger_size=int(cfg["trigger_size"]),
        trigger_margin=int(cfg["trigger_margin"]),
    )
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

    checkpoint_payload = {
        "model": model.state_dict(),
        "cfg": cfg,
        "args": vars(args),
        "configs": configs,
        "controller": controller.state_dict(),
        "final_feedback_metrics": final_feedback_metrics,
        "metrics": final_metrics,
    }
    atomic_torch_save(checkpoint_payload, os.path.join(ckpt_dir, "last.pt"))

    summary = {
        "run_name": run_name,
        "dataset": dataset_name,
        "model": model_name,
        "method": "adaptive_full",
        "adaptive_mode": args.adaptive_mode,
        "poison_rate": args.poison_rate,
        "seed": args.seed,
        "kc": kc,
        "position_mode": args.position_mode,
        "search_grid": args.search_grid,
        "search_subset": args.search_subset,
        "feedback_size": args.feedback_size,
        "feedback_source": "deterministic_subset_of_training_split",
        "target_labels": target_labels,
        "controller": controller.state_dict(),
        "final_feedback_metrics": final_feedback_metrics,
        "final_metrics": final_metrics,
        "best_feedback_metrics": best_feedback_metrics,
        "best_test_metrics_at_feedback_best": best_test_metrics,
        "elapsed_seconds": time.time() - start_time,
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))

    print("=" * 100)
    print("Final test metrics:")
    print(json.dumps(final_metrics, indent=2))
    print("Final controller weights:")
    print(json.dumps(controller.current_weights(), indent=2))
    print(f"Saved results to: {output_dir}")
    print(f"Saved checkpoint to: {ckpt_dir}/last.pt")
    print("=" * 100)


if __name__ == "__main__":
    main()
