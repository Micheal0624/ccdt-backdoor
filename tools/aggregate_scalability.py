#!/usr/bin/env python3

import ast
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable


PROJECT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT / "results"
LOG_DIR = PROJECT / "logs_extra"

RUN_PATTERN = re.compile(
    r"^(?P<dataset>cifar10|cifar100|gtsrb)_"
    r"(?P<model>resnet18|vgg11)_"
    r"full_dyn_pr0\.05_"
    r"kc(?P<kc>6|8|10)_"
    r"seed(?P<seed>0|1|2)$"
)

CORE_METRICS = [
    "clean_acc",
    "valid_asr",
    "wrong_asr",
    "single_leak",
    "invalid_leak",
    "csg",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def walk_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_dicts(value)


def normalize_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    aliases = {
        "cda": "clean_acc",
        "acc": "clean_acc",
        "clean_accuracy": "clean_acc",
        "wrong_target_asr": "wrong_asr",
        "single_trigger_leak": "single_leak",
        "single_trigger_leakage": "single_leak",
        "invalid_config_leak": "invalid_leak",
        "invalid_configuration_leakage": "invalid_leak",
    }

    output = {}

    for key, value in data.items():
        if not is_number(value):
            continue

        key = str(key).strip().lower()
        key = aliases.get(key, key)

        if (
            key in CORE_METRICS
            or re.fullmatch(r"valid_asr_c\d+", key)
        ):
            output[key] = float(value)

    return output


def metric_score(data: Dict[str, Any]) -> int:
    normalized = normalize_metrics(data)
    return sum(key in normalized for key in CORE_METRICS)


def extract_from_json(run_dir: Path):
    final_metrics = {}
    best_metrics = {}
    source = ""

    for json_path in sorted(run_dir.rglob("*.json")):
        try:
            payload = json.loads(
                json_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        all_dicts = list(walk_dicts(payload))
        final_candidates = []
        best_candidates = []

        for dictionary in all_dicts:
            for key in ("final_metrics", "final", "last_metrics"):
                value = dictionary.get(key)
                if isinstance(value, dict):
                    final_candidates.append(value)

            for key in ("best_metrics", "best"):
                value = dictionary.get(key)
                if isinstance(value, dict):
                    best_candidates.append(value)

        if not final_candidates:
            ranked = sorted(
                all_dicts,
                key=metric_score,
                reverse=True,
            )
            if ranked and metric_score(ranked[0]) >= 3:
                final_candidates.append(ranked[0])

        if final_candidates:
            candidate = max(final_candidates, key=metric_score)
            normalized = normalize_metrics(candidate)

            if normalized:
                final_metrics = normalized
                source = str(json_path)

        if best_candidates:
            candidate = max(best_candidates, key=metric_score)
            best_metrics = normalize_metrics(candidate)

        if final_metrics:
            break

    return final_metrics, best_metrics, source


def extract_dict_between(text: str, start: str, end: str):
    start_pos = text.find(start)

    if start_pos < 0:
        return {}

    start_pos += len(start)
    end_pos = text.find(end, start_pos)

    if end_pos < 0:
        block = text[start_pos:]
    else:
        block = text[start_pos:end_pos]

    left = block.find("{")
    right = block.rfind("}")

    if left < 0 or right < left:
        return {}

    try:
        value = ast.literal_eval(block[left:right + 1])
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def extract_from_log(dataset: str, model: str, kc: int, seed: int):
    log_path = LOG_DIR / (
        f"{dataset}_{model}_full_extra_"
        f"pr0.05_kc{kc}_seed{seed}_dynamic.log"
    )

    if not log_path.exists():
        return {}, {}, ""

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).replace("\r", "\n")

    final_raw = extract_dict_between(
        text,
        "Final metrics:",
        "Best metrics:",
    )
    best_raw = extract_dict_between(
        text,
        "Best metrics:",
        "Saved results to:",
    )

    return (
        normalize_metrics(final_raw),
        normalize_metrics(best_raw),
        str(log_path),
    )


def mean_std(values):
    values = [
        float(value)
        for value in values
        if is_number(value)
    ]

    if not values:
        return "", "", 0

    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std, len(values)


def write_csv(path: Path, rows):
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
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    if value == "" or value is None:
        return "NA"
    return f"{100 * float(value):.3f}%"


def main():
    rows = []
    detected_names = set()

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Missing: {RESULTS_DIR}")

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        match = RUN_PATTERN.match(run_dir.name)

        if not match:
            continue

        dataset = match.group("dataset")
        model = match.group("model")
        kc = int(match.group("kc"))
        seed = int(match.group("seed"))

        final_metrics, best_metrics, source = extract_from_json(run_dir)

        if not final_metrics:
            final_metrics, best_metrics, source = extract_from_log(
                dataset,
                model,
                kc,
                seed,
            )

        per_config_values = [
            value
            for key, value in final_metrics.items()
            if re.fullmatch(r"valid_asr_c\d+", key)
        ]

        row = {
            "run_name": run_dir.name,
            "dataset": dataset,
            "model": model,
            "method": "full",
            "position_mode": "dynamic",
            "poison_rate": 0.05,
            "kc": kc,
            "seed": seed,
            "metric_source": source,
        }

        for key, value in sorted(final_metrics.items()):
            row[f"final_{key}"] = value

        for key, value in sorted(best_metrics.items()):
            row[f"best_{key}"] = value

        if per_config_values:
            row["final_per_config_min"] = min(per_config_values)
            row["final_per_config_max"] = max(per_config_values)
            row["final_per_config_std"] = (
                statistics.pstdev(per_config_values)
                if len(per_config_values) > 1
                else 0.0
            )

        rows.append(row)
        detected_names.add(run_dir.name)

    expected_names = {
        f"{dataset}_{model}_full_dyn_pr0.05_kc{kc}_seed{seed}"
        for dataset in ("cifar10", "cifar100", "gtsrb")
        for model in ("resnet18", "vgg11")
        for kc in (6, 8, 10)
        for seed in (0, 1, 2)
    }

    missing = sorted(expected_names - detected_names)

    print("=" * 110)
    print("SCALABILITY RUN COLLECTION")
    print("=" * 110)
    print(f"Detected runs: {len(rows)}/54")
    print(f"Missing runs:  {len(missing)}")

    if missing:
        print("\nMissing run directories:")
        for name in missing:
            print("  ", name)

    run_csv = PROJECT / "scalability_runs.csv"
    write_csv(run_csv, rows)

    grouped_rows = []
    groups = {}

    for row in rows:
        key = (
            row["dataset"],
            row["model"],
            row["kc"],
        )
        groups.setdefault(key, []).append(row)

    metric_columns = sorted({
        key
        for row in rows
        for key in row
        if key.startswith("final_")
        and is_number(row.get(key))
    })

    for (dataset, model, kc), group_rows in sorted(groups.items()):
        grouped = {
            "dataset": dataset,
            "model": model,
            "method": "full",
            "position_mode": "dynamic",
            "poison_rate": 0.05,
            "kc": kc,
            "num_seeds": len(group_rows),
        }

        for metric in metric_columns:
            values = [
                row.get(metric)
                for row in group_rows
            ]
            mean, std, count = mean_std(values)
            grouped[f"{metric}_mean"] = mean
            grouped[f"{metric}_std"] = std
            grouped[f"{metric}_count"] = count

        grouped_rows.append(grouped)

    grouped_csv = PROJECT / "scalability_grouped.csv"
    write_csv(grouped_csv, grouped_rows)

    print()
    print("Grouped final metrics: mean ± std over three seeds")
    print("-" * 110)

    header = (
        f"{'Dataset':<10} {'Model':<10} {'Kc':>3} "
        f"{'Clean ACC':>19} {'Valid ASR':>19} "
        f"{'Wrong ASR':>19} {'Single Leak':>19} "
        f"{'Invalid Leak':>19} {'CSG':>19}"
    )
    print(header)

    for row in grouped_rows:
        def cell(metric):
            mean = row.get(f"final_{metric}_mean", "")
            std = row.get(f"final_{metric}_std", "")

            if mean == "" or std == "":
                return "NA"

            return f"{percent(mean)} ± {percent(std)}"

        print(
            f"{row['dataset']:<10} "
            f"{row['model']:<10} "
            f"{row['kc']:>3} "
            f"{cell('clean_acc'):>19} "
            f"{cell('valid_asr'):>19} "
            f"{cell('wrong_asr'):>19} "
            f"{cell('single_leak'):>19} "
            f"{cell('invalid_leak'):>19} "
            f"{cell('csg'):>19}"
        )

    print()
    print("Scalability trend by dataset/model")
    print("-" * 110)

    for dataset in ("cifar10", "cifar100", "gtsrb"):
        for model in ("resnet18", "vgg11"):
            subset = [
                row
                for row in grouped_rows
                if row["dataset"] == dataset
                and row["model"] == model
            ]

            if not subset:
                continue

            print(f"\n{dataset} / {model}")

            for row in sorted(subset, key=lambda item: item["kc"]):
                print(
                    f"  Kc={row['kc']:>2}: "
                    f"ACC={percent(row.get('final_clean_acc_mean'))}, "
                    f"ASR={percent(row.get('final_valid_asr_mean'))}, "
                    f"Wrong={percent(row.get('final_wrong_asr_mean'))}, "
                    f"Single={percent(row.get('final_single_leak_mean'))}, "
                    f"Invalid={percent(row.get('final_invalid_leak_mean'))}, "
                    f"CSG={percent(row.get('final_csg_mean'))}"
                )

    print()
    print(f"Saved: {run_csv}")
    print(f"Saved: {grouped_csv}")


if __name__ == "__main__":
    main()
