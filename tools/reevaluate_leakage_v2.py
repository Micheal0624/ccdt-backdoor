#!/usr/bin/env python3

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.datasets import get_datasets
from src.evaluate import evaluate_all
from src.models import get_model
from src.utils import get_device, set_seed


PROJECT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT / "results"
CHECKPOINT_ROOT = PROJECT / "checkpoints"

RUN_PATTERN = re.compile(
    r"^(?P<dataset>cifar10|cifar100|gtsrb)_"
    r"(?P<model>resnet18|vgg11)_"
    r"full_dyn_pr0\.05_"
    r"kc(?P<kc>2|3|4|6|8|10)_"
    r"seed(?P<seed>0|1|2)$"
)

CORE_METRICS = (
    "clean_acc",
    "valid_asr",
    "wrong_asr",
    "single_leak",
    "invalid_leak",
    "csg",
)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # 兼容较旧 PyTorch。
        return torch.load(path, map_location=device)


def atomic_save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def discover_runs(scope: str) -> List[Tuple[Path, Dict[str, Any]]]:
    runs = []

    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"结果目录不存在：{RESULTS_ROOT}")

    for result_dir in sorted(RESULTS_ROOT.iterdir()):
        if not result_dir.is_dir():
            continue

        match = RUN_PATTERN.match(result_dir.name)
        if not match:
            continue

        metadata = match.groupdict()
        metadata["kc"] = int(metadata["kc"])
        metadata["seed"] = int(metadata["seed"])

        if scope == "cifar10_kc10":
            if not (
                metadata["dataset"] == "cifar10"
                and metadata["kc"] == 10
            ):
                continue

        runs.append((result_dir, metadata))

    return runs


def get_test_loader(
    cfg: Dict[str, Any],
    batch_size_override: int,
    workers_override: int,
):
    dataset_name = str(cfg["dataset"])
    image_size = int(cfg["image_size"])

    _, test_set, num_classes = get_datasets(
        dataset_name=dataset_name,
        data_root=cfg["data_root"],
        image_size=image_size,
    )

    batch_size = (
        batch_size_override
        if batch_size_override > 0
        else int(cfg["batch_size"])
    )

    num_workers = (
        workers_override
        if workers_override >= 0
        else int(cfg["num_workers"])
    )

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader, num_classes


def evaluate_one(
    result_dir: Path,
    metadata: Dict[str, Any],
    checkpoint_kind: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    run_name = result_dir.name
    checkpoint_path = (
        CHECKPOINT_ROOT
        / run_name
        / f"{checkpoint_kind}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在：{checkpoint_path}"
        )

    checkpoint = load_checkpoint(checkpoint_path, device)

    required_keys = {"model", "cfg", "configs"}
    missing_keys = required_keys - set(checkpoint)

    if missing_keys:
        raise KeyError(
            f"{checkpoint_path} 缺少字段：{sorted(missing_keys)}"
        )

    cfg = checkpoint["cfg"]
    configs = checkpoint["configs"]

    if not configs:
        raise RuntimeError(f"{run_name} 的 configs 为空")

    target_labels = [
        int(config["target"])
        for config in configs
    ]

    seed = int(metadata["seed"])
    set_seed(seed)

    test_loader, num_classes = get_test_loader(
        cfg=cfg,
        batch_size_override=batch_size,
        workers_override=num_workers,
    )

    model = get_model(
        str(cfg["model"]),
        num_classes,
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    model.eval()

    start_time = time.time()

    with torch.inference_mode():
        metrics = evaluate_all(
            model=model,
            loader=test_loader,
            configs=configs,
            target_labels=target_labels,
            device=device,
            image_size=int(cfg["image_size"]),
            trigger_size=int(cfg["trigger_size"]),
            trigger_margin=int(cfg["trigger_margin"]),
        )

    elapsed = time.time() - start_time

    output = {
        "run_name": run_name,
        "dataset": metadata["dataset"],
        "model": metadata["model"],
        "kc": int(metadata["kc"]),
        "seed": seed,
        "poison_rate": 0.05,
        "position_mode": "dynamic",
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": str(checkpoint_path),
        "leakage_metric_version": 2,
        "target_labels": target_labels,
        "num_configs": len(configs),
        "elapsed_seconds": elapsed,
        "original_checkpoint_metrics": checkpoint.get("metrics", {}),
        "metrics_v2": metrics,
    }

    return output


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]

    if not numeric:
        return "", "", 0

    mean = statistics.mean(numeric)
    std = (
        statistics.stdev(numeric)
        if len(numeric) > 1
        else 0.0
    )

    return mean, std, len(numeric)


def aggregate(
    checkpoint_kind: str,
    scope: str,
) -> None:
    rows = []

    for result_dir, metadata in discover_runs(scope):
        reevaluation_path = (
            result_dir
            / f"reeval_leakage_v2_{checkpoint_kind}.json"
        )

        if not reevaluation_path.exists():
            continue

        try:
            payload = json.loads(
                reevaluation_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            print(
                f"[WARN] 无法读取 {reevaluation_path}: {exc}"
            )
            continue

        metrics = payload.get("metrics_v2", {})

        row = {
            "run_name": payload["run_name"],
            "dataset": payload["dataset"],
            "model": payload["model"],
            "kc": int(payload["kc"]),
            "seed": int(payload["seed"]),
            "checkpoint_kind": checkpoint_kind,
            "elapsed_seconds": payload.get(
                "elapsed_seconds",
                "",
            ),
        }

        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)) or value is None:
                row[key] = value

        rows.append(row)

    suffix = f"{scope}_{checkpoint_kind}"

    run_csv = (
        PROJECT
        / f"reeval_v2_{suffix}_runs.csv"
    )
    grouped_csv = (
        PROJECT
        / f"reeval_v2_{suffix}_grouped.csv"
    )

    write_csv(run_csv, rows)

    groups = {}

    for row in rows:
        group_key = (
            row["dataset"],
            row["model"],
            row["kc"],
        )
        groups.setdefault(group_key, []).append(row)

    grouped_rows = []

    for (dataset, model, kc), group_rows in sorted(groups.items()):
        grouped = {
            "dataset": dataset,
            "model": model,
            "kc": kc,
            "num_seeds": len(group_rows),
            "checkpoint_kind": checkpoint_kind,
        }

        metric_names = sorted({
            key
            for row in group_rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in {
                "kc",
                "seed",
                "elapsed_seconds",
            }
        })

        for metric_name in metric_names:
            mean, std, count = mean_std(
                row.get(metric_name)
                for row in group_rows
            )

            grouped[f"{metric_name}_mean"] = mean
            grouped[f"{metric_name}_std"] = std
            grouped[f"{metric_name}_count"] = count

        grouped_rows.append(grouped)

    write_csv(grouped_csv, grouped_rows)

    print()
    print("=" * 112)
    print("RE-EVALUATION V2 SUMMARY")
    print("=" * 112)
    print(f"Scope: {scope}")
    print(f"Checkpoint: {checkpoint_kind}")
    print(f"Detected reevaluated runs: {len(rows)}")
    print()

    header = (
        f"{'Dataset':<10} {'Model':<10} {'Kc':>3} "
        f"{'Seeds':>5} {'Valid ASR':>14} "
        f"{'Wrong ASR':>14} {'Single Leak':>14} "
        f"{'Invalid Leak':>14} {'CSG':>14}"
    )
    print(header)
    print("-" * len(header))

    def format_metric(grouped, metric):
        mean = grouped.get(f"{metric}_mean", "")
        std = grouped.get(f"{metric}_std", "")

        if mean == "" or std == "":
            return "NA"

        return (
            f"{100 * float(mean):.3f}%"
            f"±{100 * float(std):.3f}%"
        )

    for grouped in grouped_rows:
        print(
            f"{grouped['dataset']:<10} "
            f"{grouped['model']:<10} "
            f"{grouped['kc']:>3} "
            f"{grouped['num_seeds']:>5} "
            f"{format_metric(grouped, 'valid_asr'):>14} "
            f"{format_metric(grouped, 'wrong_asr'):>14} "
            f"{format_metric(grouped, 'single_leak'):>14} "
            f"{format_metric(grouped, 'invalid_leak'):>14} "
            f"{format_metric(grouped, 'csg'):>14}"
        )

    print()
    print(f"Saved: {run_csv}")
    print(f"Saved: {grouped_csv}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scope",
        choices=["cifar10_kc10", "scalability"],
        default="cifar10_kc10",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["last", "best"],
        default="last",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 表示使用配置文件中的 batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="-1 表示使用配置文件中的 num_workers",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    runs = discover_runs(args.scope)

    if args.limit is not None:
        runs = runs[
            args.start:
            args.start + args.limit
        ]
    else:
        runs = runs[args.start:]

    print(f"Scope: {args.scope}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Jobs selected: {len(runs)}")
    print(f"Resume: {args.resume}")

    if not args.aggregate_only:
        failures = []

        for index, (result_dir, metadata) in enumerate(
            runs,
            start=1,
        ):
            output_path = (
                result_dir
                / (
                    "reeval_leakage_v2_"
                    f"{args.checkpoint}.json"
                )
            )

            print()
            print("=" * 112)
            print(
                f"[{index}/{len(runs)}] "
                f"{result_dir.name}"
            )
            print("=" * 112)

            if args.resume and output_path.exists():
                print(
                    f"[SKIP] 已存在：{output_path}"
                )
                continue

            try:
                payload = evaluate_one(
                    result_dir=result_dir,
                    metadata=metadata,
                    checkpoint_kind=args.checkpoint,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

                atomic_save_json(payload, output_path)

                metrics = payload["metrics_v2"]

                print(
                    "[OK] "
                    f"ASR={metrics['valid_asr']:.6f} "
                    f"Wrong={metrics['wrong_asr']:.6f} "
                    f"Single={metrics['single_leak']:.6f} "
                    f"Invalid={metrics['invalid_leak']:.6f} "
                    f"CSG={metrics['csg']:.6f} "
                    f"Time={payload['elapsed_seconds']:.1f}s"
                )
                print(f"Saved: {output_path}")

            except Exception as exc:
                print(
                    f"[FAILED] {result_dir.name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                failures.append(
                    (result_dir.name, repr(exc))
                )

        print()
        print("=" * 112)
        print(f"Finished. Failures: {len(failures)}")

        for item in failures:
            print("[FAILED]", item)

    aggregate(
        checkpoint_kind=args.checkpoint,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
