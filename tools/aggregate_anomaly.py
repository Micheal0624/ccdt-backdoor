#!/usr/bin/env python3

import ast
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROJECT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT / "results"
LOG_DIR = PROJECT / "logs_extra"

RUN_PATTERN = re.compile(
    r"^gtsrb_(?P<model>resnet18|vgg11)_full_dyn_"
    r"pr0\.1_kc(?P<kc>3|4)_seed(?P<seed>[0-2])$"
)

METRIC_KEYS = {
    "clean_acc",
    "valid_asr",
    "wrong_asr",
    "single_leak",
    "invalid_leak",
    "csg",
}


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


def score_metric_dict(data: Dict[str, Any]) -> int:
    keys = {str(k).lower() for k in data.keys()}
    return len(keys & METRIC_KEYS)


def normalize_metrics(data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not data:
        return {}

    aliases = {
        "cda": "clean_acc",
        "acc": "clean_acc",
        "clean_accuracy": "clean_acc",
        "valid_asr": "valid_asr",
        "wrong_asr": "wrong_asr",
        "wrong_target_asr": "wrong_asr",
        "singleleak": "single_leak",
        "single_trigger_leak": "single_leak",
        "single_trigger_leakage": "single_leak",
        "invalidleak": "invalid_leak",
        "invalid_config_leak": "invalid_leak",
        "invalid_configuration_leakage": "invalid_leak",
        "csg": "csg",
    }

    output: Dict[str, float] = {}

    for key, value in data.items():
        if not is_number(value):
            continue

        normalized_key = str(key).strip().lower()
        normalized_key = aliases.get(normalized_key, normalized_key)

        if (
            normalized_key in METRIC_KEYS
            or normalized_key.startswith("valid_asr_c")
            or normalized_key.startswith("asr_c")
        ):
            output[normalized_key] = float(value)

    return output


def extract_from_json(run_dir: Path):
    final_metrics = {}
    best_metrics = {}
    source = ""

    for json_path in sorted(run_dir.rglob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        final_candidates = []
        best_candidates = []

        for dictionary in walk_dicts(payload):
            for key in ("final_metrics", "final", "last_metrics"):
                value = dictionary.get(key)
                if isinstance(value, dict):
                    final_candidates.append(value)

            for key in ("best_metrics", "best"):
                value = dictionary.get(key)
                if isinstance(value, dict):
                    best_candidates.append(value)

        if not final_candidates:
            all_dicts = list(walk_dicts(payload))
            all_dicts.sort(key=score_metric_dict, reverse=True)
            if all_dicts and score_metric_dict(all_dicts[0]) >= 3:
                final_candidates.append(all_dicts[0])

        if final_candidates:
            candidate = max(final_candidates, key=score_metric_dict)
            normalized = normalize_metrics(candidate)
            if normalized:
                final_metrics = normalized
                source = str(json_path)

        if best_candidates:
            candidate = max(best_candidates, key=score_metric_dict)
            normalized = normalize_metrics(candidate)
            if normalized:
                best_metrics = normalized

        if final_metrics:
            break

    return final_metrics, best_metrics, source


def extract_dict_between(text: str, start: str, end: Optional[str]) -> Dict[str, Any]:
    start_pos = text.find(start)
    if start_pos < 0:
        return {}

    start_pos += len(start)

    if end:
        end_pos = text.find(end, start_pos)
        block = text[start_pos:end_pos if end_pos >= 0 else None]
    else:
        block = text[start_pos:]

    left = block.find("{")
    right = block.rfind("}")

    if left < 0 or right < left:
        return {}

    try:
        value = ast.literal_eval(block[left:right + 1].strip())
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def extract_from_log(model: str, kc: int, seed: int):
    log_path = (
        LOG_DIR
        / f"gtsrb_{model}_full_extra_pr0.1_kc{kc}_seed{seed}_dynamic.log"
    )

    if not log_path.exists():
        return {}, {}, ""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r", "\n")

    final_raw = extract_dict_between(text, "Final metrics:", "Best metrics:")
    best_raw = extract_dict_between(text, "Best metrics:", "Saved results to:")

    return (
        normalize_metrics(final_raw),
        normalize_metrics(best_raw),
        str(log_path),
    )


def mean_std(values):
    values = [float(v) for v in values if is_number(v)]

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

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value == "" or value is None:
        return "NA"
    return f"{float(value):.6f}"


def main():
    run_rows = []

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        match = RUN_PATTERN.match(run_dir.name)
        if not match:
            continue

        model = match.group("model")
        kc = int(match.group("kc"))
        seed = int(match.group("seed"))

        final_metrics, best_metrics, source = extract_from_json(run_dir)

        if not final_metrics:
            final_metrics, best_metrics, source = extract_from_log(
                model=model,
                kc=kc,
                seed=seed,
            )

        row = {
            "run_name": run_dir.name,
            "dataset": "gtsrb",
            "model": model,
            "method": "full",
            "position_mode": "dynamic",
            "poison_rate": 0.1,
            "kc": kc,
            "seed": seed,
            "metric_source": source,
        }

        for key, value in sorted(final_metrics.items()):
            row[f"final_{key}"] = value

        for key, value in sorted(best_metrics.items()):
            row[f"best_{key}"] = value

        run_rows.append(row)

    expected = 12
    print("=" * 90)
    print("ANOMALY RUN COLLECTION")
    print("=" * 90)
    print(f"Detected runs: {len(run_rows)}/{expected}")

    if len(run_rows) != expected:
        print("[WARNING] 未检测到完整的 12 个实验，请检查 results 目录。")

    run_csv = PROJECT / "anomaly_runs.csv"
    write_csv(run_csv, run_rows)

    grouped_rows = []
    metric_columns = sorted(
        {
            key
            for row in run_rows
            for key in row
            if key.startswith("final_") or key.startswith("best_")
        }
    )

    groups = {}

    for row in run_rows:
        group_key = (row["model"], row["kc"])
        groups.setdefault(group_key, []).append(row)

    for (model, kc), rows in sorted(groups.items()):
        grouped = {
            "dataset": "gtsrb",
            "model": model,
            "method": "full",
            "position_mode": "dynamic",
            "poison_rate": 0.1,
            "kc": kc,
            "num_seeds": len(rows),
        }

        for metric in metric_columns:
            values = [row.get(metric) for row in rows]
            mean, std, count = mean_std(values)
            grouped[f"{metric}_mean"] = mean
            grouped[f"{metric}_std"] = std
            grouped[f"{metric}_count"] = count

        grouped_rows.append(grouped)

    grouped_csv = PROJECT / "anomaly_grouped.csv"
    write_csv(grouped_csv, grouped_rows)

    print()
    print("Per-run final metrics")
    print("-" * 90)

    header = (
        f"{'Model':<10} {'Kc':>3} {'Seed':>5} "
        f"{'CleanACC':>10} {'ValidASR':>10} {'WrongASR':>10} "
        f"{'SingleLeak':>11} {'InvalidLeak':>12} {'CSG':>10}"
    )
    print(header)

    for row in run_rows:
        print(
            f"{row['model']:<10} "
            f"{row['kc']:>3} "
            f"{row['seed']:>5} "
            f"{fmt(row.get('final_clean_acc')):>10} "
            f"{fmt(row.get('final_valid_asr')):>10} "
            f"{fmt(row.get('final_wrong_asr')):>10} "
            f"{fmt(row.get('final_single_leak')):>11} "
            f"{fmt(row.get('final_invalid_leak')):>12} "
            f"{fmt(row.get('final_csg')):>10}"
        )

    print()
    print("Grouped final metrics: mean ± std over seeds")
    print("-" * 90)

    for row in grouped_rows:
        print(f"{row['model']} | Kc={row['kc']} | seeds={row['num_seeds']}")

        for metric in (
            "final_clean_acc",
            "final_valid_asr",
            "final_wrong_asr",
            "final_single_leak",
            "final_invalid_leak",
            "final_csg",
        ):
            mean = row.get(f"{metric}_mean", "")
            std = row.get(f"{metric}_std", "")
            print(f"  {metric:<20}: {fmt(mean)} ± {fmt(std)}")

    print()
    print(f"Saved: {run_csv}")
    print(f"Saved: {grouped_csv}")


if __name__ == "__main__":
    main()
