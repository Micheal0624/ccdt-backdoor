import argparse
import copy
import csv
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets import get_datasets
from src.models import get_model
from src.evaluate import evaluate_all
from src.triggers import apply_dual_trigger
from src.poison import CCDTPoisonedDataset, make_poison_index_splits


def poison_str(x):
    return f"{float(x):g}"


def auc_rank(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda z: z[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    rank_sum = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += i

    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def tpr_at_fpr(scores, labels, max_fpr=0.05):
    # Larger score = more suspicious.
    pairs = sorted(zip(scores, labels), key=lambda z: z[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    tp = 0
    fp = 0
    best_tpr = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        if fpr <= max_fpr:
            best_tpr = max(best_tpr, tp / n_pos)
    return best_tpr


def entropy_from_logits(logits):
    probs = F.softmax(logits, dim=1).clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=1)


def make_run_name(dataset, model, method, poison_rate, kc, seed):
    return f"{dataset}_{model}_{method}_dyn_pr{poison_str(poison_rate)}_kc{kc}_seed{seed}"


def load_attack(args, device):
    run_name = make_run_name(args.dataset, args.model, args.method, args.poison_rate, args.kc, args.seed)
    ckpt_path = Path("checkpoints") / run_name / f"{args.ckpt_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt["cfg"]
    model = get_model(cfg["model"], int(cfg["num_classes"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    configs = ckpt["configs"]
    target_labels = [int(c["target"]) for c in configs]
    base_metrics = ckpt.get("metrics", {})

    return run_name, ckpt_path, ckpt, cfg, model, configs, target_labels, base_metrics


def get_loaders(cfg, batch_size, max_test=None):
    train_set, test_set, num_classes = get_datasets(
        cfg["dataset"],
        cfg["data_root"],
        image_size=int(cfg.get("image_size", 32)),
    )

    if max_test is not None and max_test > 0:
        test_indices = list(range(min(max_test, len(test_set))))
        test_eval_set = Subset(test_set, test_indices)
    else:
        test_eval_set = test_set

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_eval_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_set, test_set, train_loader, test_loader, num_classes


@torch.no_grad()
def run_strip(model, test_set, configs, cfg, device, args):
    rng = random.Random(args.seed + 12345)

    n = min(args.strip_samples, len(test_set))
    indices = list(range(len(test_set)))
    rng.shuffle(indices)
    indices = indices[:n]

    clean_scores = []
    trigger_scores = []

    image_size = int(cfg.get("image_size", 32))
    trigger_size = int(cfg.get("trigger_size", 4))
    trigger_margin = int(cfg.get("trigger_margin", 2))

    pool_size = min(max(args.strip_samples * 2, 256), len(test_set))
    pool_indices = list(range(len(test_set)))
    rng.shuffle(pool_indices)
    pool_indices = pool_indices[:pool_size]
    pool_imgs = []
    for idx in pool_indices:
        x, _ = test_set[idx]
        pool_imgs.append(x)
    pool = torch.stack(pool_imgs, dim=0).to(device)

    for local_i, idx in enumerate(indices):
        x, _ = test_set[idx]
        x = x.unsqueeze(0).to(device)

        cfg_i = configs[local_i % len(configs)]
        x_trig = apply_dual_trigger(
            x.clone(),
            cfg_i["p1"],
            cfg_i["p2"],
            image_size=image_size,
            trigger_size=trigger_size,
            margin=trigger_margin,
        )

        clean_ent = []
        trig_ent = []

        for _ in range(args.strip_repeats):
            ridx = rng.randrange(pool.shape[0])
            r = pool[ridx:ridx + 1]

            clean_mix = (args.strip_alpha * x + (1.0 - args.strip_alpha) * r).clamp(0, 1)
            trig_mix = (args.strip_alpha * x_trig + (1.0 - args.strip_alpha) * r).clamp(0, 1)

            clean_ent.append(float(entropy_from_logits(model(clean_mix)).item()))
            trig_ent.append(float(entropy_from_logits(model(trig_mix)).item()))

        clean_scores.append(sum(clean_ent) / len(clean_ent))
        trigger_scores.append(sum(trig_ent) / len(trig_ent))

    scores = [-x for x in clean_scores] + [-x for x in trigger_scores]
    labels = [0] * len(clean_scores) + [1] * len(trigger_scores)

    return {
        "strip_clean_entropy_mean": sum(clean_scores) / len(clean_scores),
        "strip_trigger_entropy_mean": sum(trigger_scores) / len(trigger_scores),
        "strip_auc": auc_rank(scores, labels),
        "strip_tpr_at_fpr_5": tpr_at_fpr(scores, labels, 0.05),
        "strip_samples": n,
        "strip_repeats": args.strip_repeats,
    }


def run_neural_cleanse(model, test_set, cfg, target_labels, device, args):
    rng = random.Random(args.seed + 999)
    n = min(args.nc_samples, len(test_set))
    indices = list(range(len(test_set)))
    rng.shuffle(indices)
    indices = indices[:n]

    xs = []
    for idx in indices:
        x, _ = test_set[idx]
        xs.append(x)
    x_batch = torch.stack(xs, dim=0).to(device)

    num_classes = int(cfg["num_classes"])
    image_size = int(cfg.get("image_size", 32))

    norms = []
    asrs = []

    model.eval()

    for target in range(num_classes):
        mask_param = torch.full((1, 1, image_size, image_size), -4.0, device=device, requires_grad=True)
        pattern_param = torch.zeros((1, 3, image_size, image_size), device=device, requires_grad=True)

        opt = torch.optim.Adam([mask_param, pattern_param], lr=args.nc_lr)

        y_target = torch.full((x_batch.shape[0],), target, dtype=torch.long, device=device)

        for _ in range(args.nc_steps):
            mask = torch.sigmoid(mask_param)
            pattern = torch.sigmoid(pattern_param)
            x_adv = (1 - mask) * x_batch + mask * pattern

            logits = model(x_adv)
            ce = F.cross_entropy(logits, y_target)
            l1 = mask.mean()
            loss = ce + args.nc_l1 * l1

            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            mask = torch.sigmoid(mask_param)
            pattern = torch.sigmoid(pattern_param)
            x_adv = (1 - mask) * x_batch + mask * pattern
            pred = model(x_adv).argmax(dim=1)
            asr = (pred == target).float().mean().item()
            norm = mask.sum().item()

        norms.append(float(norm))
        asrs.append(float(asr))

    sorted_norms = sorted(norms)
    median = sorted_norms[len(sorted_norms) // 2]
    deviations = [abs(x - median) for x in norms]
    mad = sorted(deviations)[len(deviations) // 2]
    min_norm = min(norms)
    suspicious_target = int(norms.index(min_norm))
    anomaly_index = (median - min_norm) / (mad + 1e-8)

    true_targets = set(int(t) for t in target_labels)

    return {
        "nc_anomaly_index": float(anomaly_index),
        "nc_suspicious_target": suspicious_target,
        "nc_min_mask_norm": float(min_norm),
        "nc_median_mask_norm": float(median),
        "nc_detects_true_target": int(suspicious_target in true_targets),
        "nc_suspicious_target_asr": float(asrs[suspicious_target]),
        "nc_steps": args.nc_steps,
        "nc_samples": n,
    }


def get_last_linear_module(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("No nn.Linear module found for feature extraction.")
    return last


@torch.no_grad()
def extract_features(model, loader, device):
    module = get_last_linear_module(model)
    feats = []

    def hook(_module, inputs, _output):
        x = inputs[0]
        feats.append(x.detach().flatten(1).cpu())

    handle = module.register_forward_hook(hook)

    ys = []
    for x, y in loader:
        x = x.to(device)
        _ = model(x)
        ys.append(y.cpu())

    handle.remove()

    return torch.cat(feats, dim=0), torch.cat(ys, dim=0)


def run_spectral_signatures(model, train_set, configs, cfg, device, args):
    """
    Spectral Signatures defense evaluation.

    Important:
    Build an explicit clean+poisoned mixed evaluation set.
    The previous implementation sampled the first N records from CCDTPoisonedDataset and
    could accidentally produce poison-only labels, making ROC/AUC undefined.
    """
    poison_splits = make_poison_index_splits(
        train_set,
        float(args.poison_rate),
        configs,
        int(args.seed),
    )

    poisoned_train = CCDTPoisonedDataset(
        train_set,
        configs,
        poison_splits,
        method=args.method,
        image_size=int(cfg.get("image_size", 32)),
        trigger_size=int(cfg.get("trigger_size", 4)),
        trigger_margin=int(cfg.get("trigger_margin", 2)),
    )

    records = getattr(poisoned_train, "records", None)

    def _rec_get(rec, key, default=None):
        if isinstance(rec, dict):
            return rec.get(key, default)
        return getattr(rec, key, default)

    def _is_poison_record(rec):
        # Prefer explicit boolean fields if present.
        for key in ["is_poison", "poisoned", "is_backdoor", "triggered"]:
            val = _rec_get(rec, key, None)
            if val is not None:
                return int(bool(val))

        mode = str(_rec_get(rec, "mode", "clean")).lower().strip()

        # Clean samples should be negative class.
        clean_modes = {
            "clean", "benign", "normal", "original", "none", ""
        }

        # Any trigger-bearing / injected training sample is treated as positive
        # for defense-detection evaluation, including valid dual positives,
        # single-trigger negatives, and invalid-configuration negatives.
        poison_modes = {
            "poison", "poisoned", "valid", "valid_dual",
            "dual", "single", "single_trigger",
            "invalid", "invalid_config", "invalid_configuration",
            "backdoor", "trigger", "triggered"
        }

        if mode in clean_modes:
            return 0
        if mode in poison_modes:
            return 1

        # Unknown non-empty mode is conservatively treated as poison because it
        # is usually an injected CCDT training record.
        return 1

    poison_indices = []
    if records is not None:
        # Records are expected to align with poisoned_train indexing.
        m = min(len(records), len(poisoned_train))
        for idx in range(m):
            if _is_poison_record(records[idx]):
                poison_indices.append(idx)

    n_total = int(min(args.ss_samples, len(train_set) + len(poison_indices)))
    if n_total <= 1:
        return {
            "ss_samples": n_total,
            "ss_remove_fraction": args.ss_remove_fraction,
            "ss_flagged_fraction": 0.0,
            "ss_error": "not_enough_samples",
            "ss_available_poison_records": int(len(poison_indices)),
            "ss_available_clean_records": int(len(train_set)),
        }

    desired_poison = int(round(n_total * float(args.poison_rate)))
    if float(args.poison_rate) > 0 and len(poison_indices) > 0:
        desired_poison = max(1, desired_poison)

    # Keep both classes present whenever possible.
    desired_poison = min(desired_poison, len(poison_indices), n_total - 1)
    desired_clean = min(n_total - desired_poison, len(train_set))

    # Fill remaining capacity with clean samples first, then poison samples.
    remaining = n_total - desired_clean - desired_poison
    if remaining > 0:
        add_clean = min(remaining, len(train_set) - desired_clean)
        desired_clean += add_clean
        remaining -= add_clean
    if remaining > 0:
        add_poison = min(remaining, len(poison_indices) - desired_poison)
        desired_poison += add_poison
        remaining -= add_poison

    if desired_poison <= 0 or desired_clean <= 0:
        return {
            "ss_samples": int(desired_clean + desired_poison),
            "ss_remove_fraction": args.ss_remove_fraction,
            "ss_flagged_fraction": 0.0,
            "ss_error": "missing_positive_or_negative_class",
            "ss_available_poison_records": int(len(poison_indices)),
            "ss_available_clean_records": int(len(train_set)),
            "ss_mix_clean": int(desired_clean),
            "ss_mix_poison": int(desired_poison),
        }

    g = torch.Generator()
    g.manual_seed(int(args.seed) + 20240531)

    def _sample_from_pool(pool, k):
        if k <= 0:
            return []
        if k >= len(pool):
            return list(pool)
        perm = torch.randperm(len(pool), generator=g)[:k].tolist()
        return [pool[i] for i in perm]

    clean_pool = list(range(len(train_set)))
    clean_sel = _sample_from_pool(clean_pool, desired_clean)
    poison_sel = _sample_from_pool(poison_indices, desired_poison)

    # item format: (source, index, poison_label)
    # source 0 -> original clean train_set
    # source 1 -> CCDTPoisonedDataset
    items = [(0, idx, 0) for idx in clean_sel] + [(1, idx, 1) for idx in poison_sel]
    order = torch.randperm(len(items), generator=g).tolist()
    items = [items[i] for i in order]

    class _MixedSpectralDataset(torch.utils.data.Dataset):
        def __init__(self, clean_ds, poison_ds, items):
            self.clean_ds = clean_ds
            self.poison_ds = poison_ds
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            src, idx, _flag = self.items[i]
            if src == 0:
                return self.clean_ds[idx]
            return self.poison_ds[idx]

    mixed = _MixedSpectralDataset(train_set, poisoned_train, items)
    loader = DataLoader(mixed, batch_size=args.batch_size, shuffle=False, num_workers=2)

    feats, ys = extract_features(model, loader, device)
    poison_flags = [int(x[2]) for x in items]

    scores = torch.zeros(feats.shape[0])
    flagged = torch.zeros(feats.shape[0], dtype=torch.bool)

    removal_fraction = args.ss_remove_fraction
    for cls in sorted(set(int(x) for x in ys.tolist())):
        idxs = (ys == cls).nonzero(as_tuple=False).flatten()
        if idxs.numel() < 5:
            continue

        X = feats[idxs].float()
        X = X - X.mean(dim=0, keepdim=True)

        try:
            _, _, vh = torch.linalg.svd(X, full_matrices=False)
            v = vh[0]
            cls_scores = (X @ v).pow(2)
        except Exception:
            continue

        scores[idxs] = cls_scores.cpu()

        k = max(1, int(math.ceil(removal_fraction * idxs.numel())))
        top_local = torch.topk(cls_scores, k=k, largest=True).indices
        flagged[idxs[top_local]] = True

    labels = poison_flags
    score_list = [float(x) for x in scores.tolist()]
    pred = flagged.tolist()

    tp = sum(1 for p, y in zip(pred, labels) if p and y == 1)
    fp = sum(1 for p, y in zip(pred, labels) if p and y == 0)
    fn = sum(1 for p, y in zip(pred, labels) if (not p) and y == 1)
    tn = sum(1 for p, y in zip(pred, labels) if (not p) and y == 0)

    return {
        "ss_samples": int(len(items)),
        "ss_remove_fraction": float(removal_fraction),
        "ss_flagged_fraction": float(flagged.float().mean().item()),
        "ss_available_poison_records": int(len(poison_indices)),
        "ss_available_clean_records": int(len(train_set)),
        "ss_mix_clean": int(desired_clean),
        "ss_mix_poison": int(desired_poison),
        "ss_auc": auc_rank(score_list, labels),
        "ss_tpr_at_fpr_5": tpr_at_fpr(score_list, labels, 0.05),
        "ss_precision": tp / max(tp + fp, 1),
        "ss_recall": tp / max(tp + fn, 1),
        "ss_poison_rate_observed": sum(labels) / max(len(labels), 1),
        "ss_tp": int(tp),
        "ss_fp": int(fp),
        "ss_fn": int(fn),
        "ss_tn": int(tn),
    }


def get_last_conv_module(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No nn.Conv2d module found for fine-pruning.")
    return last


@torch.no_grad()
def compute_channel_activation(model, loader, device, max_batches):
    module = get_last_conv_module(model)
    sums = []
    count = 0

    def hook(_module, _inputs, output):
        val = output.detach().abs().mean(dim=(0, 2, 3)).cpu()
        sums.append(val)

    handle = module.register_forward_hook(hook)

    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        _ = model(x)
        count += 1

    handle.remove()

    if not sums:
        raise RuntimeError("No activations collected for pruning.")

    return torch.stack(sums, dim=0).mean(dim=0)


def run_fine_pruning(model, test_loader, configs, target_labels, cfg, device, args):
    image_size = int(cfg.get("image_size", 32))
    trigger_size = int(cfg.get("trigger_size", 4))
    trigger_margin = int(cfg.get("trigger_margin", 2))

    # Use clean test loader for channel ranking.
    act = compute_channel_activation(model, test_loader, device, args.fp_rank_batches)
    order = torch.argsort(act)  # low activation first

    module = get_last_conv_module(model)
    n_ch = int(act.numel())

    rows = []

    for frac in args.fp_fractions:
        prune_n = int(round(frac * n_ch))
        mask = torch.ones(n_ch, device=device)
        if prune_n > 0:
            mask[order[:prune_n].to(device)] = 0.0

        def prune_hook(_module, _inputs, output):
            return output * mask.view(1, -1, 1, 1)

        handle = module.register_forward_hook(prune_hook)

        metrics = evaluate_all(
            model,
            test_loader,
            configs,
            target_labels,
            device,
            image_size=image_size,
            trigger_size=trigger_size,
            trigger_margin=trigger_margin,
        )

        handle.remove()

        row = {
            "fp_fraction": float(frac),
            "fp_pruned_channels": prune_n,
            "fp_total_channels": n_ch,
        }
        for k, v in metrics.items():
            row[f"fp_{k}"] = float(v)
        rows.append(row)

    best = min(rows, key=lambda r: r.get("fp_valid_asr", 1e9))
    out = {}
    for k, v in best.items():
        out[f"best_{k}"] = v

    # Also keep a compact trajectory string for inspection.
    out["fp_curve"] = json.dumps(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--poison-rate", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kc", type=int, default=2)
    ap.add_argument("--defense", required=True, choices=["strip", "neural_cleanse", "spectral_signatures", "fine_pruning"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-name", default="last", choices=["best", "last"])

    ap.add_argument("--strip-samples", type=int, default=300)
    ap.add_argument("--strip-repeats", type=int, default=10)
    ap.add_argument("--strip-alpha", type=float, default=0.5)

    ap.add_argument("--nc-samples", type=int, default=128)
    ap.add_argument("--nc-steps", type=int, default=60)
    ap.add_argument("--nc-lr", type=float, default=0.1)
    ap.add_argument("--nc-l1", type=float, default=0.01)

    ap.add_argument("--ss-samples", type=int, default=5000)
    ap.add_argument("--ss-remove-fraction", type=float, default=0.1)

    ap.add_argument("--fp-rank-batches", type=int, default=20)
    ap.add_argument("--fp-fractions", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])

    args = ap.parse_args()

    device = torch.device(args.device)
    run_name, ckpt_path, ckpt, cfg, model, configs, target_labels, base_metrics = load_attack(args, device)

    train_set, test_set, train_loader, test_loader, _ = get_loaders(cfg, args.batch_size)

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "method": args.method,
        "poison_rate": args.poison_rate,
        "seed": args.seed,
        "kc": args.kc,
        "defense": args.defense,
        "run_name": run_name,
        "checkpoint": str(ckpt_path),
    }

    for k, v in base_metrics.items():
        if isinstance(v, (int, float)):
            result[f"base_{k}"] = float(v)

    if args.defense == "strip":
        result.update(run_strip(model, test_set, configs, cfg, device, args))
    elif args.defense == "neural_cleanse":
        result.update(run_neural_cleanse(model, test_set, cfg, target_labels, device, args))
    elif args.defense == "spectral_signatures":
        result.update(run_spectral_signatures(model, train_set, configs, cfg, device, args))
    elif args.defense == "fine_pruning":
        result.update(run_fine_pruning(model, test_loader, configs, target_labels, cfg, device, args))
    else:
        raise ValueError(args.defense)

    out_dir = Path("results/defenses") / run_name / args.defense
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 100)
    print("Saved:", out_path)
    print(json.dumps(result, indent=2))
    print("=" * 100)


if __name__ == "__main__":
    main()
