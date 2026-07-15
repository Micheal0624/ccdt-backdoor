#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from src.datasets import get_datasets
from src.models import get_model
from src.triggers import apply_dual_trigger
from src.utils import get_device, set_seed


PROJECT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT / "results"
CHECKPOINT_ROOT = PROJECT / "checkpoints"

RUN_PATTERN = re.compile(
    r"^(?P<dataset>cifar10|cifar100|gtsrb)_"
    r"(?P<model>resnet18|vgg11)_"
    r"full_dyn_pr0\.05_"
    r"kc(?P<kc>4|8)_"
    r"seed(?P<seed>0|1|2)$"
)

CATEGORIES = (
    "seen_p1_unseen_p2",
    "unseen_p1_seen_p2",
    "unseen_p1_unseen_p2",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        choices=["last", "best"],
        default="last",
    )
    parser.add_argument(
        "--pairs-per-category",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="full",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
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
        "--aggregate-only",
        action="store_true",
    )

    return parser.parse_args()


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def atomic_save_json(
    payload: Dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(path)


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
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
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def discover_runs(
    requested_run_name: str = "",
) -> List[Tuple[Path, Dict[str, Any]]]:
    runs = []

    for result_dir in sorted(RESULTS_ROOT.iterdir()):
        if not result_dir.is_dir():
            continue

        if requested_run_name:
            if result_dir.name != requested_run_name:
                continue

        match = RUN_PATTERN.match(result_dir.name)

        if not match:
            continue

        metadata = match.groupdict()
        metadata["kc"] = int(metadata["kc"])
        metadata["seed"] = int(metadata["seed"])

        runs.append((result_dir, metadata))

    return runs


def position_xy(position) -> Tuple[int, int]:
    if isinstance(position, dict):
        return int(position["y"]), int(position["x"])

    if isinstance(position, (list, tuple)):
        return int(position[0]), int(position[1])

    raise TypeError(
        f"Unsupported position type: {type(position).__name__}"
    )


def position_name(position) -> str:
    y, x = position_xy(position)
    return f"y{y}_x{x}"


def make_position(y: int, x: int) -> Dict[str, Any]:
    return {
        "type": "xy",
        "y": int(y),
        "x": int(x),
        "name": f"y{int(y)}_x{int(x)}",
    }


def normalize_checkpoint_args(value) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    if hasattr(value, "__dict__"):
        return vars(value)

    return {}


def build_grid_positions(
    cfg: Dict[str, Any],
    checkpoint_args: Dict[str, Any],
    kc: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    image_size = int(cfg["image_size"])
    trigger_size = int(cfg["trigger_size"])
    margin = int(cfg["trigger_margin"])

    grid_size = int(
        checkpoint_args.get(
            "search_grid",
            6 if kc >= 6 else 4,
        )
    )

    if grid_size < 2:
        raise ValueError(
            f"search_grid must be >= 2, got {grid_size}"
        )

    low = margin
    high = image_size - margin - trigger_size

    axis = [
        int(round(
            low + index * (high - low) / (grid_size - 1)
        ))
        for index in range(grid_size)
    ]

    axis = list(dict.fromkeys(axis))

    positions = [
        make_position(y, x)
        for y in axis
        for x in axis
    ]

    return axis, positions


def deterministic_sample(
    items: List[Dict[str, Any]],
    limit: int,
    key: str,
) -> List[Dict[str, Any]]:
    if limit <= 0 or limit >= len(items):
        return list(items)

    def score(item):
        text = (
            f"{key}|"
            f"{item['category']}|"
            f"{position_name(item['p1'])}|"
            f"{position_name(item['p2'])}"
        )

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    return sorted(items, key=score)[:limit]


def build_out_of_library_pairs(
    run_name: str,
    configs: List[Dict[str, Any]],
    grid_positions: List[Dict[str, Any]],
    pairs_per_category: int,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    used_coordinates = {
        position_xy(config["p1"])
        for config in configs
    } | {
        position_xy(config["p2"])
        for config in configs
    }

    unused_positions = [
        position
        for position in grid_positions
        if position_xy(position) not in used_coordinates
    ]

    if len(unused_positions) < 2:
        raise RuntimeError(
            f"{run_name}: 未使用网格位置不足，"
            f"unused={len(unused_positions)}"
        )

    seen_p1 = [
        {
            "position": config["p1"],
            "config_id": int(config["config_id"]),
        }
        for config in configs
    ]

    seen_p2 = [
        {
            "position": config["p2"],
            "config_id": int(config["config_id"]),
        }
        for config in configs
    ]

    candidates = {
        category: []
        for category in CATEGORIES
    }

    for item in seen_p1:
        for unseen in unused_positions:
            candidates["seen_p1_unseen_p2"].append({
                "category": "seen_p1_unseen_p2",
                "p1": item["position"],
                "p2": unseen,
                "seen_config_id": item["config_id"],
            })

    for unseen in unused_positions:
        for item in seen_p2:
            candidates["unseen_p1_seen_p2"].append({
                "category": "unseen_p1_seen_p2",
                "p1": unseen,
                "p2": item["position"],
                "seen_config_id": item["config_id"],
            })

    for first in unused_positions:
        for second in unused_positions:
            if position_xy(first) == position_xy(second):
                continue

            candidates["unseen_p1_unseen_p2"].append({
                "category": "unseen_p1_unseen_p2",
                "p1": first,
                "p2": second,
                "seen_config_id": None,
            })

    selected = {}

    for category in CATEGORIES:
        selected[category] = deterministic_sample(
            candidates[category],
            pairs_per_category,
            f"{run_name}:{category}",
        )

        for pair_index, pair in enumerate(
            selected[category],
            start=1,
        ):
            pair["pair_id"] = (
                f"{category}_{pair_index:02d}"
            )

    return (
        selected,
        sorted(used_coordinates),
        sorted(position_xy(p) for p in unused_positions),
    )


def make_target_mask(
    predictions: torch.Tensor,
    target_labels: List[int],
) -> torch.Tensor:
    mask = torch.zeros_like(
        predictions,
        dtype=torch.bool,
    )

    for target in target_labels:
        mask |= predictions == int(target)

    return mask


def percentile(
    values: List[float],
    quantile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0

    mean = statistics.mean(values)

    std = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    return mean, std


def get_test_loader(
    cfg: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    max_test_samples: int,
):
    _, test_set, num_classes = get_datasets(
        dataset_name=str(cfg["dataset"]),
        data_root=cfg["data_root"],
        image_size=int(cfg["image_size"]),
    )

    if max_test_samples > 0:
        sample_count = min(
            int(max_test_samples),
            len(test_set),
        )

        test_set = Subset(
            test_set,
            list(range(sample_count)),
        )

    loader = DataLoader(
        test_set,
        batch_size=(
            batch_size
            if batch_size > 0
            else int(cfg["batch_size"])
        ),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader, num_classes, len(test_set)


def collect_clean_predictions(
    model,
    loader,
    device,
) -> torch.Tensor:
    predictions = []

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            predictions.append(
                model(images)
                .argmax(dim=1)
                .cpu()
            )

    if not predictions:
        raise RuntimeError("测试集为空")

    return torch.cat(predictions, dim=0)


def evaluate_pair(
    model,
    loader,
    clean_predictions: torch.Tensor,
    pair: Dict[str, Any],
    target_labels: List[int],
    device: torch.device,
    image_size: int,
    trigger_size: int,
    trigger_margin: int,
) -> Dict[str, Any]:
    total = 0
    changed_count = 0
    target_count = 0
    leakage_count = 0

    target_changed_counts = {
        int(target): 0
        for target in target_labels
    }

    offset = 0

    with torch.inference_mode():
        for images, _ in loader:
            batch_size = images.size(0)

            clean_batch = clean_predictions[
                offset:offset + batch_size
            ].to(
                device,
                non_blocking=True,
            )

            offset += batch_size

            images = images.to(
                device,
                non_blocking=True,
            )

            triggered = apply_dual_trigger(
                images,
                pair["p1"],
                pair["p2"],
                image_size=image_size,
                trigger_size=trigger_size,
                margin=trigger_margin,
            )

            predictions = model(triggered).argmax(dim=1)

            changed = predictions != clean_batch
            in_target_set = make_target_mask(
                predictions,
                target_labels,
            )
            leakage = changed & in_target_set

            total += batch_size
            changed_count += changed.sum().item()
            target_count += in_target_set.sum().item()
            leakage_count += leakage.sum().item()

            for target in target_labels:
                target_changed_counts[int(target)] += (
                    leakage
                    & (predictions == int(target))
                ).sum().item()

    target_changed_rates = {
        str(target): count / max(total, 1)
        for target, count in target_changed_counts.items()
    }

    worst_target = max(
        target_changed_counts,
        key=target_changed_counts.get,
    )

    return {
        "pair_id": pair["pair_id"],
        "category": pair["category"],
        "seen_config_id": pair.get("seen_config_id"),
        "p1": {
            "name": position_name(pair["p1"]),
            "y": position_xy(pair["p1"])[0],
            "x": position_xy(pair["p1"])[1],
        },
        "p2": {
            "name": position_name(pair["p2"]),
            "y": position_xy(pair["p2"])[0],
            "x": position_xy(pair["p2"])[1],
        },
        "total": total,
        "prediction_change_count": changed_count,
        "prediction_change_rate": (
            changed_count / max(total, 1)
        ),
        "raw_target_count": target_count,
        "raw_target_rate": (
            target_count / max(total, 1)
        ),
        "target_leakage_count": leakage_count,
        "target_leakage_rate": (
            leakage_count / max(total, 1)
        ),
        "target_changed_counts": target_changed_counts,
        "target_changed_rates": target_changed_rates,
        "worst_target": int(worst_target),
        "worst_target_changed_rate": (
            target_changed_counts[worst_target]
            / max(total, 1)
        ),
    }


def summarize_pairs(
    pair_results: List[Dict[str, Any]],
    target_labels: List[int],
) -> Dict[str, Any]:
    total_instances = sum(
        int(row["total"])
        for row in pair_results
    )

    total_changed = sum(
        int(row["prediction_change_count"])
        for row in pair_results
    )

    total_raw_target = sum(
        int(row["raw_target_count"])
        for row in pair_results
    )

    total_leakage = sum(
        int(row["target_leakage_count"])
        for row in pair_results
    )

    leakages = [
        float(row["target_leakage_rate"])
        for row in pair_results
    ]

    changes = [
        float(row["prediction_change_rate"])
        for row in pair_results
    ]

    raw_targets = [
        float(row["raw_target_rate"])
        for row in pair_results
    ]

    mean_leakage, std_leakage = mean_std(leakages)
    mean_change, std_change = mean_std(changes)
    mean_raw_target, std_raw_target = mean_std(raw_targets)

    worst_pair = max(
        pair_results,
        key=lambda row: row["target_leakage_rate"],
    )

    pooled_target_counts = {
        int(target): 0
        for target in target_labels
    }

    for row in pair_results:
        for target, count in row[
            "target_changed_counts"
        ].items():
            pooled_target_counts[int(target)] += int(count)

    worst_target = max(
        pooled_target_counts,
        key=pooled_target_counts.get,
    )

    return {
        "num_pairs": len(pair_results),
        "total_pair_sample_instances": total_instances,
        "pooled_prediction_change_rate": (
            total_changed / max(total_instances, 1)
        ),
        "pooled_raw_target_rate": (
            total_raw_target / max(total_instances, 1)
        ),
        "pooled_target_leakage_rate": (
            total_leakage / max(total_instances, 1)
        ),
        "mean_pair_target_leakage": mean_leakage,
        "std_pair_target_leakage": std_leakage,
        "median_pair_target_leakage": percentile(
            leakages,
            0.50,
        ),
        "p95_pair_target_leakage": percentile(
            leakages,
            0.95,
        ),
        "max_pair_target_leakage": max(leakages),
        "mean_pair_prediction_change": mean_change,
        "std_pair_prediction_change": std_change,
        "max_pair_prediction_change": max(changes),
        "mean_pair_raw_target_rate": mean_raw_target,
        "std_pair_raw_target_rate": std_raw_target,
        "max_pair_raw_target_rate": max(raw_targets),
        "worst_pair_id": worst_pair["pair_id"],
        "worst_pair_p1": worst_pair["p1"]["name"],
        "worst_pair_p2": worst_pair["p2"]["name"],
        "worst_target": int(worst_target),
        "worst_target_changed_rate": (
            pooled_target_counts[worst_target]
            / max(total_instances, 1)
        ),
        "pooled_target_changed_counts": (
            pooled_target_counts
        ),
    }


def evaluate_one(
    result_dir: Path,
    metadata: Dict[str, Any],
    args,
    device: torch.device,
) -> Dict[str, Any]:
    run_name = result_dir.name

    checkpoint_path = (
        CHECKPOINT_ROOT
        / run_name
        / f"{args.checkpoint}.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint 不存在: {checkpoint_path}"
        )

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    required = {"model", "cfg", "configs"}
    missing = required - set(checkpoint)

    if missing:
        raise KeyError(
            f"{checkpoint_path} 缺少字段: {sorted(missing)}"
        )

    cfg = checkpoint["cfg"]
    configs = checkpoint["configs"]
    checkpoint_args = normalize_checkpoint_args(
        checkpoint.get("args", {})
    )

    target_labels = [
        int(config["target"])
        for config in configs
    ]

    set_seed(int(metadata["seed"]))

    grid_axis, grid_positions = build_grid_positions(
        cfg,
        checkpoint_args,
        int(metadata["kc"]),
    )

    (
        selected_pairs,
        used_coordinates,
        unused_coordinates,
    ) = build_out_of_library_pairs(
        run_name=run_name,
        configs=configs,
        grid_positions=grid_positions,
        pairs_per_category=args.pairs_per_category,
    )

    loader, num_classes, test_size = get_test_loader(
        cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_test_samples=args.max_test_samples,
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

    clean_predictions = collect_clean_predictions(
        model,
        loader,
        device,
    )

    clean_target_mask = make_target_mask(
        clean_predictions,
        target_labels,
    )

    category_results = {}
    all_pair_results = []

    total_pairs = sum(
        len(value)
        for value in selected_pairs.values()
    )

    completed_pairs = 0

    for category in CATEGORIES:
        category_pair_results = []

        for pair in selected_pairs[category]:
            completed_pairs += 1

            result = evaluate_pair(
                model=model,
                loader=loader,
                clean_predictions=clean_predictions,
                pair=pair,
                target_labels=target_labels,
                device=device,
                image_size=int(cfg["image_size"]),
                trigger_size=int(cfg["trigger_size"]),
                trigger_margin=int(cfg["trigger_margin"]),
            )

            category_pair_results.append(result)
            all_pair_results.append(result)

            print(
                f"  [PAIR {completed_pairs}/{total_pairs}] "
                f"{category} "
                f"{result['p1']['name']}+"
                f"{result['p2']['name']} "
                f"Leak={result['target_leakage_rate']:.6f} "
                f"Change={result['prediction_change_rate']:.6f}",
                flush=True,
            )

        category_results[category] = {
            "summary": summarize_pairs(
                category_pair_results,
                target_labels,
            ),
            "pairs": category_pair_results,
        }

    output = {
        "experiment": "out_of_library_invalid",
        "definition": (
            "Dual-trigger configurations containing at least "
            "one search-grid position that was not selected into "
            "the model configuration library during training."
        ),
        "run_name": run_name,
        "dataset": metadata["dataset"],
        "model": metadata["model"],
        "kc": int(metadata["kc"]),
        "seed": int(metadata["seed"]),
        "position_mode": "dynamic",
        "checkpoint_kind": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "tag": args.tag,
        "test_size": test_size,
        "pairs_per_category_requested": (
            args.pairs_per_category
        ),
        "target_labels": target_labels,
        "search_grid": int(
            checkpoint_args.get(
                "search_grid",
                6 if int(metadata["kc"]) >= 6 else 4,
            )
        ),
        "grid_axis": grid_axis,
        "grid_position_count": len(grid_positions),
        "used_position_count": len(used_coordinates),
        "unused_position_count": len(unused_coordinates),
        "used_positions": [
            {"y": y, "x": x, "name": f"y{y}_x{x}"}
            for y, x in used_coordinates
        ],
        "unused_positions": [
            {"y": y, "x": x, "name": f"y{y}_x{x}"}
            for y, x in unused_coordinates
        ],
        "clean_target_prediction_rate": (
            clean_target_mask.sum().item()
            / max(test_size, 1)
        ),
        "overall": summarize_pairs(
            all_pair_results,
            target_labels,
        ),
        "categories": category_results,
        "elapsed_seconds": time.time() - start_time,
    }

    return output


def aggregate(args) -> None:
    run_rows = []
    category_rows = []
    pair_rows = []

    filename = (
        f"out_of_library_invalid_"
        f"{args.tag}_{args.checkpoint}.json"
    )

    for result_dir, metadata in discover_runs():
        path = result_dir / filename

        if not path.is_file():
            continue

        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception as error:
            print(
                f"[WARN] 无法读取 {path}: {error}",
                flush=True,
            )
            continue

        overall = payload["overall"]

        run_rows.append({
            "run_name": payload["run_name"],
            "dataset": payload["dataset"],
            "model": payload["model"],
            "kc": payload["kc"],
            "seed": payload["seed"],
            "test_size": payload["test_size"],
            "num_pairs": overall["num_pairs"],
            "clean_target_prediction_rate": (
                payload["clean_target_prediction_rate"]
            ),
            "pooled_target_leakage_rate": (
                overall["pooled_target_leakage_rate"]
            ),
            "mean_pair_target_leakage": (
                overall["mean_pair_target_leakage"]
            ),
            "p95_pair_target_leakage": (
                overall["p95_pair_target_leakage"]
            ),
            "max_pair_target_leakage": (
                overall["max_pair_target_leakage"]
            ),
            "pooled_prediction_change_rate": (
                overall["pooled_prediction_change_rate"]
            ),
            "pooled_raw_target_rate": (
                overall["pooled_raw_target_rate"]
            ),
            "worst_target_changed_rate": (
                overall["worst_target_changed_rate"]
            ),
            "elapsed_seconds": payload["elapsed_seconds"],
        })

        for category, content in payload[
            "categories"
        ].items():
            summary = content["summary"]

            category_rows.append({
                "run_name": payload["run_name"],
                "dataset": payload["dataset"],
                "model": payload["model"],
                "kc": payload["kc"],
                "seed": payload["seed"],
                "category": category,
                **{
                    key: value
                    for key, value in summary.items()
                    if isinstance(value, (int, float, str))
                },
            })

            for pair in content["pairs"]:
                pair_rows.append({
                    "run_name": payload["run_name"],
                    "dataset": payload["dataset"],
                    "model": payload["model"],
                    "kc": payload["kc"],
                    "seed": payload["seed"],
                    "category": category,
                    "pair_id": pair["pair_id"],
                    "p1": pair["p1"]["name"],
                    "p2": pair["p2"]["name"],
                    "seen_config_id": pair[
                        "seen_config_id"
                    ],
                    "prediction_change_rate": pair[
                        "prediction_change_rate"
                    ],
                    "raw_target_rate": pair[
                        "raw_target_rate"
                    ],
                    "target_leakage_rate": pair[
                        "target_leakage_rate"
                    ],
                    "worst_target": pair[
                        "worst_target"
                    ],
                    "worst_target_changed_rate": pair[
                        "worst_target_changed_rate"
                    ],
                })

    suffix = f"{args.tag}_{args.checkpoint}"

    write_csv(
        PROJECT
        / f"out_of_library_invalid_{suffix}_runs.csv",
        run_rows,
    )

    write_csv(
        PROJECT
        / f"out_of_library_invalid_{suffix}_categories.csv",
        category_rows,
    )

    write_csv(
        PROJECT
        / f"out_of_library_invalid_{suffix}_pairs.csv",
        pair_rows,
    )

    grouped_rows = []

    group_keys = sorted({
        (
            row["dataset"],
            row["model"],
            int(row["kc"]),
            row["category"],
        )
        for row in category_rows
    })

    metrics = [
        "pooled_target_leakage_rate",
        "mean_pair_target_leakage",
        "p95_pair_target_leakage",
        "max_pair_target_leakage",
        "pooled_prediction_change_rate",
        "pooled_raw_target_rate",
        "worst_target_changed_rate",
    ]

    for dataset, model, kc, category in group_keys:
        group = [
            row
            for row in category_rows
            if row["dataset"] == dataset
            and row["model"] == model
            and int(row["kc"]) == kc
            and row["category"] == category
        ]

        grouped = {
            "dataset": dataset,
            "model": model,
            "kc": kc,
            "category": category,
            "num_seeds": len(group),
        }

        for metric in metrics:
            values = [
                float(row[metric])
                for row in group
            ]

            mean, std = mean_std(values)

            grouped[f"{metric}_mean"] = mean
            grouped[f"{metric}_std"] = std

        grouped_rows.append(grouped)

    write_csv(
        PROJECT
        / f"out_of_library_invalid_{suffix}_grouped.csv",
        grouped_rows,
    )

    print()
    print("=" * 142)
    print("OUT-OF-LIBRARY INVALID CONFIGURATION SUMMARY")
    print("=" * 142)
    print(
        f"Detected completed runs: {len(run_rows)}/36"
    )
    print(
        f"Category records: {len(category_rows)}/108"
    )
    print()

    print(
        f"{'Dataset':<10} "
        f"{'Model':<10} "
        f"{'Kc':>3} "
        f"{'Category':<25} "
        f"{'Seeds':>5} "
        f"{'Leakage':>18} "
        f"{'P95':>18} "
        f"{'Max-pair':>18} "
        f"{'Change':>18}"
    )
    print("-" * 142)

    for row in grouped_rows:
        print(
            f"{row['dataset']:<10} "
            f"{row['model']:<10} "
            f"{int(row['kc']):>3} "
            f"{row['category']:<25} "
            f"{int(row['num_seeds']):>5} "
            f"{row['pooled_target_leakage_rate_mean']*100:>8.3f}%"
            f"±{row['pooled_target_leakage_rate_std']*100:<7.3f}% "
            f"{row['p95_pair_target_leakage_mean']*100:>8.3f}%"
            f"±{row['p95_pair_target_leakage_std']*100:<7.3f}% "
            f"{row['max_pair_target_leakage_mean']*100:>8.3f}%"
            f"±{row['max_pair_target_leakage_std']*100:<7.3f}% "
            f"{row['pooled_prediction_change_rate_mean']*100:>8.3f}%"
            f"±{row['pooled_prediction_change_rate_std']*100:<7.3f}%"
        )

    print()
    print(
        "Saved:",
        PROJECT
        / f"out_of_library_invalid_{suffix}_runs.csv",
    )
    print(
        "Saved:",
        PROJECT
        / f"out_of_library_invalid_{suffix}_categories.csv",
    )
    print(
        "Saved:",
        PROJECT
        / f"out_of_library_invalid_{suffix}_pairs.csv",
    )
    print(
        "Saved:",
        PROJECT
        / f"out_of_library_invalid_{suffix}_grouped.csv",
    )


def main():
    args = parse_args()
    device = get_device()

    runs = discover_runs(args.run_name)

    start = max(0, int(args.start))
    selected = runs[start:]

    if args.limit is not None:
        selected = selected[:max(0, int(args.limit))]

    print("=" * 100)
    print("Out-of-Library Invalid Configuration Evaluation")
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Tag:", args.tag)
    print("Discovered runs:", len(runs))
    print("Selected runs:", len(selected))
    print("Pairs/category:", args.pairs_per_category)
    print("Max test samples:", args.max_test_samples)
    print("=" * 100, flush=True)

    if args.aggregate_only:
        aggregate(args)
        return

    output_filename = (
        f"out_of_library_invalid_"
        f"{args.tag}_{args.checkpoint}.json"
    )

    failures = []

    for index, (result_dir, metadata) in enumerate(
        selected,
        start=1,
    ):
        output_path = result_dir / output_filename

        print()
        print("=" * 100)
        print(
            f"[RUN {index}/{len(selected)}] "
            f"{result_dir.name}"
        )
        print("=" * 100, flush=True)

        if args.resume and output_path.is_file():
            print(
                f"[SKIP] 已存在: {output_path}",
                flush=True,
            )
            continue

        try:
            payload = evaluate_one(
                result_dir=result_dir,
                metadata=metadata,
                args=args,
                device=device,
            )

            atomic_save_json(
                payload,
                output_path,
            )

            print(
                f"[OK] {result_dir.name} "
                f"Leak="
                f"{payload['overall']['pooled_target_leakage_rate']:.6f} "
                f"Max="
                f"{payload['overall']['max_pair_target_leakage']:.6f} "
                f"Time={payload['elapsed_seconds']:.1f}s",
                flush=True,
            )

        except Exception as error:
            print(
                f"[FAILED] {result_dir.name}: {error}",
                flush=True,
            )
            traceback.print_exc()
            failures.append(result_dir.name)
            break

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate(args)

    print()
    print("=" * 100)
    print("Finished")
    print("Failures:", len(failures))

    for run_name in failures:
        print("FAILED:", run_name)

    print("=" * 100)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
