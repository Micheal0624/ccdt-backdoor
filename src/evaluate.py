from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .triggers import apply_trigger, apply_dual_trigger


@torch.no_grad()
def evaluate_clean(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_ccdt(
    model: torch.nn.Module,
    loader: DataLoader,
    configs: List[Dict],
    target_labels: List[int],
    device: torch.device,
    image_size: int = 32,
    trigger_size: int = 4,
    trigger_margin: int = 2,
) -> Dict[str, float]:
    model.eval()

    target_set = set(int(t) for t in target_labels)

    def prediction_in_target_set(pred: torch.Tensor) -> torch.Tensor:
        """Return whether each prediction belongs to any attack target."""
        hit = torch.zeros_like(pred, dtype=torch.bool)
        for target_value in target_set:
            hit |= pred == int(target_value)
        return hit

    def label_outside_target_set(labels: torch.Tensor) -> torch.Tensor:
        """Legacy eligibility mask used only for diagnostic reporting."""
        mask = torch.ones_like(labels, dtype=torch.bool)
        for target_value in target_set:
            mask &= labels != int(target_value)
        return mask

    # Cache the deterministic test batches and their clean predictions once.
    # This avoids recomputing the clean model output for every trigger layout.
    eval_batches = []

    for x_cpu, y_cpu in loader:
        x_cpu = x_cpu.detach().cpu()
        y_cpu = y_cpu.detach().cpu()

        clean_pred_cpu = (
            model(x_cpu.to(device, non_blocking=True))
            .argmax(dim=1)
            .detach()
            .cpu()
        )

        eval_batches.append((x_cpu, y_cpu, clean_pred_cpu))

    if not eval_batches:
        raise RuntimeError("Evaluation loader is empty.")

    valid_asrs = []
    wrong_asrs = []

    # Valid-configuration ASR and wrong-target ASR.
    for cfg in configs:
        target = int(cfg["target"])
        total = 0
        hit_target = 0
        hit_wrong_target = 0

        for x_cpu, y_cpu, _ in eval_batches:
            mask = y_cpu != target

            if mask.sum().item() == 0:
                continue

            x = x_cpu[mask].to(device, non_blocking=True)
            y = y_cpu[mask].to(device, non_blocking=True)

            x_bd = apply_dual_trigger(
                x,
                cfg["p1"],
                cfg["p2"],
                image_size,
                trigger_size,
                trigger_margin,
            )

            pred = model(x_bd).argmax(dim=1)

            total += y.numel()
            hit_target += (pred == target).sum().item()

            wrong_targets = [
                int(t)
                for t in target_labels
                if int(t) != target
            ]

            if wrong_targets:
                wrong_mask = torch.zeros_like(
                    pred,
                    dtype=torch.bool,
                )

                for wrong_target in wrong_targets:
                    wrong_mask |= pred == wrong_target

                hit_wrong_target += wrong_mask.sum().item()

        valid_asrs.append(hit_target / max(total, 1))
        wrong_asrs.append(hit_wrong_target / max(total, 1))

    # ------------------------------------------------------------------
    # Universal single-trigger leakage.
    #
    # A sample leaks when:
    #   1. the triggered prediction differs from its clean prediction; and
    #   2. the new prediction belongs to the attack-target set.
    #
    # This remains defined even when target_labels cover every class.
    # ------------------------------------------------------------------
    single_total = 0
    single_hit = 0

    # Preserve the old conditional statistic for diagnosis and comparison.
    single_legacy_total = 0
    single_legacy_hit = 0

    for cfg in configs:
        trigger_positions = [
            (1, cfg["p1"]),
            (2, cfg["p2"]),
        ]

        for trigger_id, position in trigger_positions:
            for x_cpu, y_cpu, clean_pred_cpu in eval_batches:
                x = x_cpu.to(device, non_blocking=True)

                x_bd = apply_trigger(
                    x,
                    trigger_id,
                    position,
                    image_size,
                    trigger_size,
                    trigger_margin,
                )

                pred_cpu = (
                    model(x_bd)
                    .argmax(dim=1)
                    .detach()
                    .cpu()
                )

                in_target = prediction_in_target_set(pred_cpu)
                changed = pred_cpu != clean_pred_cpu
                universal_hit = in_target & changed

                single_total += y_cpu.numel()
                single_hit += universal_hit.sum().item()

                legacy_mask = label_outside_target_set(y_cpu)
                single_legacy_total += legacy_mask.sum().item()

                if legacy_mask.any():
                    single_legacy_hit += (
                        in_target[legacy_mask]
                        .sum()
                        .item()
                    )

    # ------------------------------------------------------------------
    # Universal invalid-configuration leakage.
    #
    # An invalid pair leaks when it changes the clean prediction into any
    # attack target. This is well-defined for Kc equal to the class count.
    # ------------------------------------------------------------------
    invalid_total = 0
    invalid_hit = 0

    invalid_legacy_total = 0
    invalid_legacy_hit = 0

    for cfg_a in configs:
        for cfg_b in configs:
            if int(cfg_a["config_id"]) == int(cfg_b["config_id"]):
                continue

            for x_cpu, y_cpu, clean_pred_cpu in eval_batches:
                x = x_cpu.to(device, non_blocking=True)

                x_bd = apply_dual_trigger(
                    x,
                    cfg_a["p1"],
                    cfg_b["p2"],
                    image_size,
                    trigger_size,
                    trigger_margin,
                )

                pred_cpu = (
                    model(x_bd)
                    .argmax(dim=1)
                    .detach()
                    .cpu()
                )

                in_target = prediction_in_target_set(pred_cpu)
                changed = pred_cpu != clean_pred_cpu
                universal_hit = in_target & changed

                invalid_total += y_cpu.numel()
                invalid_hit += universal_hit.sum().item()

                legacy_mask = label_outside_target_set(y_cpu)
                invalid_legacy_total += legacy_mask.sum().item()

                if legacy_mask.any():
                    invalid_legacy_hit += (
                        in_target[legacy_mask]
                        .sum()
                        .item()
                    )

    if single_total == 0:
        raise RuntimeError(
            "Universal single-trigger evaluation has zero samples."
        )

    if invalid_total == 0:
        raise RuntimeError(
            "Universal invalid-configuration evaluation has zero samples."
        )

    valid_asr = sum(valid_asrs) / max(len(valid_asrs), 1)
    wrong_asr = max(wrong_asrs) if wrong_asrs else 0.0

    single_leak = single_hit / single_total
    invalid_leak = invalid_hit / invalid_total

    single_leak_legacy = (
        single_legacy_hit / single_legacy_total
        if single_legacy_total > 0
        else None
    )

    invalid_leak_legacy = (
        invalid_legacy_hit / invalid_legacy_total
        if invalid_legacy_total > 0
        else None
    )

    csg = valid_asr - max(
        wrong_asr,
        single_leak,
        invalid_leak,
    )

    result = {
        "valid_asr": valid_asr,
        "wrong_asr": wrong_asr,

        # Universal leakage metrics used by the revised CSG.
        "single_leak": single_leak,
        "invalid_leak": invalid_leak,
        "csg": csg,

        # Metric definition/version metadata.
        "leakage_metric_version": 2,

        # Raw counts make empty-set errors immediately visible.
        "single_hit": int(single_hit),
        "single_total": int(single_total),
        "invalid_hit": int(invalid_hit),
        "invalid_total": int(invalid_total),

        # Original conditional metrics retained only for comparison.
        "single_leak_legacy": single_leak_legacy,
        "invalid_leak_legacy": invalid_leak_legacy,
        "single_legacy_total": int(single_legacy_total),
        "invalid_legacy_total": int(invalid_legacy_total),
    }

    for i, value in enumerate(valid_asrs):
        result[f"valid_asr_c{i + 1}"] = value

    return result


@torch.no_grad()
def evaluate_all(
    model: torch.nn.Module,
    loader: DataLoader,
    configs: List[Dict],
    target_labels: List[int],
    device: torch.device,
    image_size: int = 32,
    trigger_size: int = 4,
    trigger_margin: int = 2,
) -> Dict[str, float]:
    clean_acc = evaluate_clean(model, loader, device)
    ccdt = evaluate_ccdt(
        model=model,
        loader=loader,
        configs=configs,
        target_labels=target_labels,
        device=device,
        image_size=image_size,
        trigger_size=trigger_size,
        trigger_margin=trigger_margin,
    )
    ccdt["clean_acc"] = clean_acc
    return ccdt
