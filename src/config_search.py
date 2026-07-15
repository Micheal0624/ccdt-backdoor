import math
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


def _make_position(y: int, x: int) -> Dict:
    return {
        "type": "xy",
        "y": int(y),
        "x": int(x),
        "name": f"y{int(y)}_x{int(x)}",
    }


def build_candidate_positions(
    image_size: int,
    trigger_size: int,
    margin: int,
    grid_size: int,
) -> List[Dict]:
    low = margin
    high = image_size - margin - trigger_size

    if grid_size <= 1:
        coords = [low]
    else:
        coords = [
            int(round(low + i * (high - low) / (grid_size - 1)))
            for i in range(grid_size)
        ]

    candidates = []
    seen = set()

    for y in coords:
        for x in coords:
            key = (y, x)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(_make_position(y, x))

    return candidates


def _patch_variance(
    x: torch.Tensor,
    y: int,
    x0: int,
    trigger_size: int,
) -> float:
    patch = x[:, y:y + trigger_size, x0:x0 + trigger_size]
    return float(patch.var().item())


def score_candidate_positions(
    dataset: Dataset,
    candidates: List[Dict],
    image_size: int,
    trigger_size: int,
    subset_size: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)

    n = len(dataset)
    indices = list(range(n))
    rng.shuffle(indices)
    indices = indices[:min(subset_size, n)]

    scores = {c["name"]: [] for c in candidates}

    for idx in indices:
        x, _ = dataset[idx]

        if not isinstance(x, torch.Tensor):
            raise TypeError("Dataset transform must return torch.Tensor.")

        if x.dim() != 3:
            raise ValueError(f"Expected CHW tensor, got {tuple(x.shape)}")

        for c in candidates:
            v = _patch_variance(
                x,
                y=int(c["y"]),
                x0=int(c["x"]),
                trigger_size=trigger_size,
            )
            scores[c["name"]].append(v)

    scored = []
    for c in candidates:
        vals = scores[c["name"]]
        mean_var = sum(vals) / max(len(vals), 1)

        item = dict(c)
        item["texture_score"] = float(mean_var)
        scored.append(item)

    scored.sort(key=lambda z: z["texture_score"])
    return scored


def _distance(p1: Dict, p2: Dict) -> float:
    dy = float(p1["y"] - p2["y"])
    dx = float(p1["x"] - p2["x"])
    return math.sqrt(dy * dy + dx * dx)


def _build_config_from_pair(item: Dict, cid: int, target_labels: List[int]) -> Dict:
    p1 = item["p1"]
    p2 = item["p2"]

    return {
        "config_id": cid,
        "name": f"dc{cid + 1}",
        "p1": {
            "type": "xy",
            "y": int(p1["y"]),
            "x": int(p1["x"]),
            "name": p1["name"],
            "texture_score": float(p1["texture_score"]),
        },
        "p2": {
            "type": "xy",
            "y": int(p2["y"]),
            "x": int(p2["x"]),
            "name": p2["name"],
            "texture_score": float(p2["texture_score"]),
        },
        "target": int(target_labels[cid]),
        "search_score": float(item["score"]),
        "distance_score": float(item["dist"]),
    }


def _select_configs_greedy(
    pair_items: List[Dict],
    kc: int,
    target_labels: List[int],
    strict_unique_positions: bool,
) -> List[Dict]:
    configs = []
    used_pairs = set()
    used_positions = set()

    for item in pair_items:
        p1 = item["p1"]
        p2 = item["p2"]

        key = (p1["name"], p2["name"])
        rev_key = (p2["name"], p1["name"])

        # Avoid duplicate ordered pair and reverse pair.
        if key in used_pairs or rev_key in used_pairs:
            continue

        # For Kc >= 3, avoid reusing the same physical position across configs.
        # This suppresses configuration interference such as:
        #   c1: T1@A + T2@B
        #   c3: T1@A + T2@C
        if strict_unique_positions:
            if p1["name"] in used_positions or p2["name"] in used_positions:
                continue

        cid = len(configs)
        configs.append(_build_config_from_pair(item, cid, target_labels))

        used_pairs.add(key)
        used_positions.add(p1["name"])
        used_positions.add(p2["name"])

        if len(configs) >= kc:
            break

    return configs


def search_dynamic_configs(
    dataset: Dataset,
    kc: int,
    target_labels: List[int],
    image_size: int,
    trigger_size: int,
    margin: int,
    grid_size: int,
    subset_size: int,
    seed: int,
    distance_weight: float = 0.25,
) -> List[Dict]:
    if len(target_labels) < kc:
        raise ValueError("target_labels length must be >= kc.")

    candidates = build_candidate_positions(
        image_size=image_size,
        trigger_size=trigger_size,
        margin=margin,
        grid_size=grid_size,
    )

    scored = score_candidate_positions(
        dataset=dataset,
        candidates=candidates,
        image_size=image_size,
        trigger_size=trigger_size,
        subset_size=subset_size,
        seed=seed,
    )

    max_dist = math.sqrt(2.0) * image_size

    pair_items = []

    for p1 in scored:
        for p2 in scored:
            if p1["name"] == p2["name"]:
                continue

            dist = _distance(p1, p2) / max(max_dist, 1e-8)

            # lower texture is better; larger distance is better
            pair_score = (
                -0.5 * (p1["texture_score"] + p2["texture_score"])
                + distance_weight * dist
            )

            pair_items.append({
                "p1": p1,
                "p2": p2,
                "score": float(pair_score),
                "dist": float(dist),
            })

    pair_items.sort(key=lambda z: z["score"], reverse=True)

    # Kc=2 keeps the original behavior.
    # Kc>=3 first uses diverse unique-position selection.
    if kc >= 3:
        configs = _select_configs_greedy(
            pair_items=pair_items,
            kc=kc,
            target_labels=target_labels,
            strict_unique_positions=True,
        )

        if len(configs) < kc:
            print(
                f"[WARN] Only found {len(configs)} strict-diverse dynamic configs; "
                "falling back to reverse-pair-only selection."
            )
            configs = _select_configs_greedy(
                pair_items=pair_items,
                kc=kc,
                target_labels=target_labels,
                strict_unique_positions=False,
            )
    else:
        configs = _select_configs_greedy(
            pair_items=pair_items,
            kc=kc,
            target_labels=target_labels,
            strict_unique_positions=False,
        )

    if len(configs) < kc:
        raise RuntimeError(f"Only found {len(configs)} dynamic configs, need {kc}.")

    print("=" * 80)
    print("Dynamic configuration search result:")
    if kc >= 3:
        print("Selection mode: diverse_unique_positions")
    else:
        print("Selection mode: original_pair_selection")

    used_position_names = []
    for cfg in configs:
        used_position_names.append(cfg["p1"]["name"])
        used_position_names.append(cfg["p2"]["name"])

        print(
            f"{cfg['name']}: "
            f"T1@{cfg['p1']['name']} + T2@{cfg['p2']['name']} "
            f"-> target {cfg['target']} "
            f"score={cfg['search_score']:.6f}"
        )

    if kc >= 3:
        num_unique = len(set(used_position_names))
        print(f"Unique physical positions: {num_unique}/{len(used_position_names)}")

    print("=" * 80)

    return configs
