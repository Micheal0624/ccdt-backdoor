#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHOD_ORDER = ["single", "naive_dual", "wo_invalid", "full"]
METHOD_LABEL = {
    "single": "Single",
    "naive_dual": "Naive dual",
    "wo_invalid": "w/o invalid",
    "full": "Full",
}
METHOD_COLOR = {
    "single": "#8DA0CB",
    "naive_dual": "#FC8D62",
    "wo_invalid": "#66C2A5",
    "full": "#E78AC3",
}

METRIC_LABEL = {
    "valid_asr": "Valid ASR",
    "single_leak": "Single-trigger Leak",
    "invalid_leak": "Invalid-config Leak",
    "wrong_asr": "Wrong ASR",
    "csg": "CSG",
    "strip_auc": "STRIP AUC",
    "ss_auc": "SS AUC",
    "nc_detects_true_target": "NC true-target hit",
    "delta_csg": "Fine-pruning ΔCSG",
    "delta_valid_asr": "Fine-pruning ΔValid ASR",
    "fp_csg": "Fine-pruning CSG",
    "fp_valid_asr": "Fine-pruning Valid ASR",
    "base_csg": "Base CSG",
    "base_valid_asr": "Base Valid ASR",
    "per_config_asr": "Per-config ASR",
}

METRIC_ALIASES = {
    "valid_asr": ["valid_asr", "asr_valid", "validasr", "valid_attack_success_rate"],
    "single_leak": ["single_leak", "singleleak", "single_trigger_leak", "single_trigger_leakage"],
    "invalid_leak": ["invalid_leak", "invalidleak", "invalid_config_leak", "invalid_configuration_leak"],
    "wrong_asr": ["wrong_asr", "wrongasr", "wrong_target_asr"],
    "csg": ["csg", "config_selectivity_gap", "configuration_selectivity_gap"],
    "strip_auc": ["strip_auc", "auc_strip"],
    "ss_auc": ["ss_auc", "spectral_auc", "spectral_signatures_auc"],
    "nc_detects_true_target": ["nc_detects_true_target", "nc_true_target", "neural_cleanse_true_target", "nc_hit"],
    "delta_csg": ["delta_csg", "fp_delta_csg", "fine_pruning_delta_csg"],
    "delta_valid_asr": ["delta_valid_asr", "fp_delta_valid_asr", "fine_pruning_delta_valid_asr"],
    "fp_csg": ["fp_csg", "csg_after_fp", "post_fp_csg", "fine_pruning_csg"],
    "fp_valid_asr": ["fp_valid_asr", "valid_asr_after_fp", "post_fp_valid_asr", "fine_pruning_valid_asr"],
    "base_csg": ["base_csg", "pre_fp_csg", "csg_before_fp"],
    "base_valid_asr": ["base_valid_asr", "base_asr", "pre_fp_valid_asr", "valid_asr_before_fp"],
    "per_config_asr": ["per_config_asr", "config_asr", "asr_mean", "valid_asr", "asr"],
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    v = float(m.group(0))
    if "%" in s and abs(v) > 1:
        v /= 100.0
    return v


def parse_pm_std(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    if "±" not in s and "+/-" not in s:
        return np.nan
    parts = re.split(r"±|\+/-", s)
    if len(parts) < 2:
        return np.nan
    return parse_number(parts[1])


def numeric(df, col):
    return df[col].map(parse_number).astype(float)


def col_by_alias(df, aliases, required=True):
    aliases_n = [norm(a) for a in aliases]
    nmap = {c: norm(c) for c in df.columns}

    for a in aliases_n:
        for c, n in nmap.items():
            if n == a:
                return c

    for a in aliases_n:
        for c, n in nmap.items():
            if a and a in n:
                return c

    if required:
        raise KeyError(f"Missing column aliases={aliases}. Existing={list(df.columns)}")
    return None


def id_col(df, kind, required=True):
    aliases = {
        "dataset": ["dataset", "data", "dataset_name", "ds"],
        "model": ["model", "arch", "network", "backbone"],
        "method": ["method", "attack", "variant", "training_method"],
        "poison_rate": ["poison_rate", "poisonrate", "poison", "pr", "rho"],
        "kc": ["kc", "Kc", "num_configs", "num_config", "n_config", "num_valid_configs"],
        "seed": ["seed", "run_seed", "random_seed"],
        "config_id": ["config_id", "config", "cid", "config_idx", "configuration_id"],
        "target_label": ["target_label", "target", "target_class", "target_y"],
        "positions": ["trigger_positions", "positions", "position", "trigger_position", "config_positions"],
    }
    return col_by_alias(df, aliases[kind], required=required)


def metric_cols(df, metric, required=True):
    aliases = [norm(a) for a in METRIC_ALIASES.get(metric, [metric])]
    nmap = {c: norm(c) for c in df.columns}

    mean_candidates = []
    std_candidates = []

    for c, n in nmap.items():
        if metric == "valid_asr" and ("invalid" in n or "wrong" in n):
            continue
        if metric == "csg" and "delta" in n:
            continue

        hit_score = 0
        for a in aliases:
            if n == a:
                hit_score = max(hit_score, 100)
            elif n in {a + "mean", "mean" + a, a + "avg", "avg" + a}:
                hit_score = max(hit_score, 90)
            elif n in {a + "std", "std" + a, a + "stdev", "stdev" + a}:
                hit_score = max(hit_score, 80)
            elif a in n:
                hit_score = max(hit_score, 20)

        if not hit_score:
            continue

        if "std" in n or "stdev" in n or "stderr" in n:
            std_candidates.append((hit_score, c))
        else:
            mean_candidates.append((hit_score, c))

    if not mean_candidates:
        if required:
            raise KeyError(f"Missing metric={metric}. Existing={list(df.columns)}")
        return None, None

    mean_col = sorted(mean_candidates, reverse=True)[0][1]
    std_col = sorted(std_candidates, reverse=True)[0][1] if std_candidates else None
    return mean_col, std_col


def metric_mean_std(df, metric, required=True):
    mc, sc = metric_cols(df, metric, required=required)
    if mc is None:
        return None, None

    mean = numeric(df, mc)

    if sc:
        std = numeric(df, sc)
    else:
        std = df[mc].map(parse_pm_std).astype(float)
        if std.notna().sum() == 0:
            std = None

    return mean, std


def pretty_dataset(x):
    s = str(x)
    low = s.lower()
    if low in ["cifar10", "cifar-10"]:
        return "CIFAR-10"
    if low in ["cifar100", "cifar-100"]:
        return "CIFAR-100"
    if low == "gtsrb":
        return "GTSRB"
    return s


def pretty_model(x):
    s = str(x)
    low = s.lower()
    if low in ["resnet18", "resnet-18"]:
        return "ResNet-18"
    if low in ["vgg11", "vgg-11"]:
        return "VGG11"
    return s


def setting_label(dataset, model):
    return f"{pretty_dataset(dataset)} / {pretty_model(model)}"


def method_norm(x):
    return str(x).strip()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def savefig(fig, path):
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def clean_axis(ax):
    ax.grid(axis="y", color="#CFCFCF", alpha=0.35, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_manifest(manifest, fig, src, status, note=""):
    manifest.append({
        "figure": fig,
        "source": str(src),
        "status": status,
        "note": str(note),
    })


def filter_pr(df, target=0.05):
    pr_col = id_col(df, "poison_rate", required=False)
    if pr_col is None:
        return df.copy(), None
    vals = sorted(numeric(df, pr_col).dropna().unique())
    if not vals:
        return df.copy(), None
    chosen = min(vals, key=lambda v: abs(v - target))
    return df[np.isclose(numeric(df, pr_col), chosen)].copy(), chosen


def filter_full(df):
    mcol = id_col(df, "method", required=False)
    if mcol is None:
        return df.copy()
    sub = df[df[mcol].map(method_norm) == "full"].copy()
    return sub if not sub.empty else df.copy()


def percent(v):
    return np.asarray(v, dtype=float) * 100.0


def nice_upper_percent(values, min_upper=1.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return min_upper
    mx = float(np.nanmax(vals))
    raw = max(min_upper, mx * 1.30)
    if raw <= 1:
        step = 0.2
    elif raw <= 3:
        step = 0.5
    elif raw <= 10:
        step = 1.0
    else:
        step = 5.0
    return math.ceil(raw / step) * step


def axis_percent(ax, metric, values):
    vals_pct = percent(values)

    if metric in ["valid_asr", "csg", "base_csg", "fp_csg", "base_valid_asr", "fp_valid_asr", "per_config_asr"]:
        finite = vals_pct[np.isfinite(vals_pct)]
        if finite.size and np.nanmin(finite) >= 85:
            ax.set_ylim(90, 100.5)
            ax.set_yticks([90, 95, 100])
        else:
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
    elif metric in ["single_leak", "invalid_leak", "wrong_asr"]:
        upper = nice_upper_percent(vals_pct, min_upper=1.0)
        ax.set_ylim(0, upper)
        if upper <= 1:
            ax.set_yticks([0, 0.5, 1.0])
        elif upper <= 3:
            ax.set_yticks(np.linspace(0, upper, 4))
        else:
            ax.set_yticks(np.linspace(0, upper, 5))
    elif metric in ["delta_csg", "delta_valid_asr"]:
        finite = vals_pct[np.isfinite(vals_pct)]
        if finite.size:
            mx = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1.0)
        else:
            mx = 1.0
        upper = math.ceil(mx * 1.25)
        ax.set_ylim(-upper, upper)
        ax.axhline(0, color="black", linewidth=0.8)
    else:
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])

    ax.set_ylabel(f"{METRIC_LABEL.get(metric, metric)} (%)")


def aggregate_by_method(df, metric):
    mcol = id_col(df, "method")
    mean, std = metric_mean_std(df, metric)

    rows = []
    for method in METHOD_ORDER:
        idx = df.index[df[mcol].map(method_norm) == method].tolist()
        vals = mean.loc[idx].dropna().values

        if len(vals) == 0:
            rows.append((method, np.nan, np.nan))
        else:
            avg = float(np.mean(vals))
            if std is not None and std.loc[idx].notna().any():
                err = float(np.nanmean(std.loc[idx].values))
            else:
                err = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
            rows.append((method, avg, err))

    return rows


def bar_methods(ax, rows, metric, title):
    labels = [METHOD_LABEL.get(r[0], r[0]) for r in rows]
    means = np.array([r[1] for r in rows], dtype=float)
    errs = np.array([0 if np.isnan(r[2]) else r[2] for r in rows], dtype=float)

    x = np.arange(len(rows))
    ax.bar(
        x,
        percent(means),
        width=0.62,
        color=[METHOD_COLOR.get(r[0], "#BBBBBB") for r in rows],
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    axis_percent(ax, metric, means)
    clean_axis(ax)


def agg_by_pr(df, metric):
    mean, std = metric_mean_std(df, metric, required=False)
    if mean is None:
        return None

    pr_col = id_col(df, "poison_rate", required=False)
    if pr_col is None:
        pr = pd.Series([0.0] * len(df), index=df.index)
    else:
        pr = numeric(df, pr_col)

    rows = []
    tmp = pd.DataFrame({"pr": pr, "mean": mean})
    if std is not None:
        tmp["std"] = std
    else:
        tmp["std"] = np.nan

    for pr_val, g in tmp.groupby("pr"):
        vals = g["mean"].dropna().values
        if len(vals) == 0:
            continue
        avg = float(np.mean(vals))
        if g["std"].notna().any():
            err = float(np.nanmean(g["std"].values))
        else:
            err = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
        rows.append((float(pr_val), avg, err))

    return sorted(rows, key=lambda x: x[0])


def bar_poison(ax, rows, metric, title, color="#9ECAE1", chance_line=None):
    x = np.arange(len(rows))
    means = np.array([r[1] for r in rows], dtype=float)
    errs = np.array([0 if np.isnan(r[2]) else r[2] for r in rows], dtype=float)

    ax.bar(
        x,
        percent(means),
        width=0.58,
        color=color,
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )

    if chance_line is not None:
        ax.axhline(chance_line * 100, color="gray", linestyle="--", linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]:g}" for r in rows])
    ax.set_xlabel("Poison rate")
    ax.set_title(title)
    axis_percent(ax, metric, means)
    clean_axis(ax)


def fig5_main_ablation(main_csv, out_dir, manifest):
    name = "fig5_main_ablation_pub_v5"
    try:
        df = read_csv(main_csv)
        df, pr = filter_pr(df, 0.05)

        metrics = [
            ("valid_asr", "Valid ASR"),
            ("single_leak", "Single-trigger leakage"),
            ("invalid_leak", "Invalid-configuration leakage"),
            ("csg", "Configuration selectivity gap"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
        axes = axes.ravel()

        for ax, (metric, title) in zip(axes, metrics):
            rows = aggregate_by_method(df, metric)
            bar_methods(ax, rows, metric, title)

        pr_txt = "0.05" if pr is None else f"{pr:g}"
        fig.suptitle(f"Main ablation across dataset/model settings, poison rate = {pr_txt}", fontsize=10)
        savefig(fig, Path(out_dir) / "main" / f"{name}.png")
        add_manifest(manifest, name, main_csv, "ok", f"poison_rate={pr_txt}")

    except Exception as e:
        add_manifest(manifest, name, main_csv, "failed", repr(e))


def fig6_kc(kc_csv, out_dir, manifest):
    name = "fig6_kc_scalability_pub_v5"
    try:
        df = read_csv(kc_csv)
        dcol = id_col(df, "dataset")
        mcol = id_col(df, "model")
        kcol = id_col(df, "kc")
        mean, _ = metric_mean_std(df, "csg")

        datasets = sorted(df[dcol].dropna().unique(), key=str)
        fig, axes = plt.subplots(1, len(datasets), figsize=(9.2, 3.0), sharey=True, constrained_layout=True)
        if len(datasets) == 1:
            axes = [axes]

        for ax, ds in zip(axes, datasets):
            sub = df[df[dcol] == ds].copy()
            for model in sorted(sub[mcol].dropna().unique(), key=str):
                g = sub[sub[mcol] == model].copy()
                g["_kc"] = numeric(g, kcol)
                g = g.sort_values("_kc")
                ax.plot(
                    g["_kc"],
                    percent(mean.loc[g.index]),
                    marker="o" if "resnet" in str(model).lower() else "s",
                    linewidth=1.6,
                    markersize=4.5,
                    label=pretty_model(model),
                )

            ax.set_title(pretty_dataset(ds))
            ax.set_xlabel("Kc")
            ax.set_xticks(sorted(numeric(df, kcol).dropna().unique()))
            ax.set_ylim(93, 100.5)
            ax.set_yticks([94, 96, 98, 100])
            clean_axis(ax)

        axes[0].set_ylabel("CSG (%)")
        axes[-1].legend(frameon=False, loc="lower left")
        fig.suptitle("Scalability to more valid configurations", fontsize=10)
        savefig(fig, Path(out_dir) / "main" / f"{name}.png")
        add_manifest(manifest, name, kc_csv, "ok")

    except Exception as e:
        add_manifest(manifest, name, kc_csv, "failed", repr(e))


def fig7_defense(def_csv, out_dir, manifest):
    name = "fig7_defense_summary_pub_v5"
    try:
        df = filter_full(read_csv(def_csv))

        metric_specs = [
            ("strip_auc", "STRIP", "#9ECAE1", 0.5),
            ("ss_auc", "Spectral Signatures", "#9ECAE1", 0.5),
            ("nc_detects_true_target", "Neural Cleanse", "#BCBDDC", None),
            ("delta_csg", "Fine-pruning", "#FDD0A2", None),
        ]

        if metric_mean_std(df, "delta_csg", required=False)[0] is None:
            metric_specs[-1] = ("fp_csg", "Fine-pruning", "#FDD0A2", None)

        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
        axes = axes.ravel()

        for ax, (metric, title, color, chance) in zip(axes, metric_specs):
            rows = agg_by_pr(df, metric)
            if rows is None or len(rows) == 0:
                ax.axis("off")
                ax.set_title(f"{title}: missing")
                continue
            bar_poison(ax, rows, metric, f"{title}: {METRIC_LABEL.get(metric, metric)}", color=color, chance_line=chance)

        fig.suptitle("Defense evaluation on the full method", fontsize=10)
        savefig(fig, Path(out_dir) / "main" / f"{name}.png")
        add_manifest(manifest, name, def_csv, "ok")

    except Exception as e:
        add_manifest(manifest, name, def_csv, "failed", repr(e))


def app_a1_full_poison(main_csv, out_dir, manifest):
    name = "app_a1_full_poison_sensitivity_pub_v5"
    try:
        df = read_csv(main_csv)
        mcol = id_col(df, "method")
        df = df[df[mcol].map(method_norm) == "full"].copy()

        metrics = [
            ("valid_asr", "Valid ASR"),
            ("single_leak", "Single-trigger leakage"),
            ("invalid_leak", "Invalid-configuration leakage"),
            ("csg", "CSG"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
        axes = axes.ravel()

        for ax, (metric, title) in zip(axes, metrics):
            rows = agg_by_pr(df, metric)
            bar_poison(ax, rows, metric, title, color="#B2DF8A")

        fig.suptitle("Full method across poison rates", fontsize=10)
        savefig(fig, Path(out_dir) / "appendix" / f"{name}.png")
        add_manifest(manifest, name, main_csv, "ok")

    except Exception as e:
        add_manifest(manifest, name, main_csv, "failed", repr(e))


def app_a2_per_setting(main_csv, out_dir, manifest):
    try:
        df = read_csv(main_csv)
        dcol = id_col(df, "dataset")
        mcol = id_col(df, "model")
        meth_col = id_col(df, "method")
        pr_col = id_col(df, "poison_rate")

        settings = sorted(
            df[[dcol, mcol]].drop_duplicates().itertuples(index=False, name=None),
            key=lambda x: (str(x[0]), str(x[1]))
        )
        pr_values = sorted(numeric(df, pr_col).dropna().unique())

        metrics = ["valid_asr", "single_leak", "invalid_leak", "wrong_asr", "csg"]

        for metric in metrics:
            if metric_mean_std(df, metric, required=False)[0] is None:
                add_manifest(manifest, f"app_a2_{metric}_per_setting_pub_v5", main_csv, "skipped", "metric missing")
                continue

            name = f"app_a2_{metric}_per_setting_pub_v5"
            mean, _ = metric_mean_std(df, metric)

            fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), sharey=True, constrained_layout=True)
            axes = axes.ravel()

            all_vals = []

            for ax, (ds, model) in zip(axes, settings):
                sub = df[(df[dcol] == ds) & (df[mcol] == model)].copy()
                sub["_pr"] = numeric(sub, pr_col)
                sub["_method"] = sub[meth_col].map(method_norm)
                sub["_val"] = mean.loc[sub.index]

                x = np.arange(len(pr_values))
                width = 0.18

                for j, method in enumerate(METHOD_ORDER):
                    vals = []
                    for pr in pr_values:
                        g = sub[(sub["_method"] == method) & np.isclose(sub["_pr"], pr)]
                        val = float(g["_val"].mean()) if len(g) else np.nan
                        vals.append(val)
                        if np.isfinite(val):
                            all_vals.append(val)

                    ax.bar(
                        x + (j - 1.5) * width,
                        percent(vals),
                        width=width,
                        color=METHOD_COLOR[method],
                        edgecolor="black",
                        linewidth=0.4,
                        label=METHOD_LABEL[method],
                    )

                ax.set_title(setting_label(ds, model), pad=5)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{p:g}" for p in pr_values])
                ax.set_xlabel("Poison rate")
                clean_axis(ax)

            for ax in axes[len(settings):]:
                ax.axis("off")

            for ax in axes[:len(settings)]:
                axis_percent(ax, metric, all_vals)

            axes[0].set_ylabel(f"{METRIC_LABEL.get(metric, metric)} (%)")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04))
            fig.suptitle(f"{METRIC_LABEL.get(metric, metric)} across poison rates", fontsize=10, y=1.08)

            savefig(fig, Path(out_dir) / "appendix" / f"{name}.png")
            add_manifest(manifest, name, main_csv, "ok")

    except Exception as e:
        add_manifest(manifest, "app_a2_per_setting_pub_v5", main_csv, "failed", repr(e))


def app_a3_kc(kc_csv, out_dir, manifest):
    name = "app_a3_kc_detailed_pub_v5"
    try:
        df = read_csv(kc_csv)
        dcol = id_col(df, "dataset")
        mcol = id_col(df, "model")
        kcol = id_col(df, "kc")

        datasets = sorted(df[dcol].dropna().unique(), key=str)
        metrics = ["valid_asr", "single_leak", "invalid_leak", "csg"]

        out_pdf = Path(out_dir) / "appendix" / f"{name}.pdf"
        ensure_dir(out_pdf.parent)

        first_png = False
        with PdfPages(out_pdf) as pdf:
            for metric in metrics:
                mean, _ = metric_mean_std(df, metric, required=False)
                if mean is None:
                    continue

                fig, axes = plt.subplots(1, len(datasets), figsize=(9.2, 3.0), sharey=True, constrained_layout=True)
                if len(datasets) == 1:
                    axes = [axes]

                all_vals = []

                for ax, ds in zip(axes, datasets):
                    sub = df[df[dcol] == ds].copy()
                    for model in sorted(sub[mcol].dropna().unique(), key=str):
                        g = sub[sub[mcol] == model].copy()
                        g["_kc"] = numeric(g, kcol)
                        g = g.sort_values("_kc")
                        vals = mean.loc[g.index].values
                        all_vals.extend(vals[np.isfinite(vals)])
                        ax.plot(
                            g["_kc"],
                            percent(vals),
                            marker="o",
                            linewidth=1.6,
                            markersize=4.5,
                            label=pretty_model(model),
                        )

                    ax.set_title(pretty_dataset(ds))
                    ax.set_xlabel("Kc")
                    ax.set_xticks(sorted(numeric(df, kcol).dropna().unique()))
                    clean_axis(ax)

                for ax in axes:
                    axis_percent(ax, metric, all_vals)

                axes[0].set_ylabel(f"{METRIC_LABEL[metric]} (%)")
                axes[-1].legend(frameon=False, loc="best")
                fig.suptitle(f"Kc scalability: {METRIC_LABEL[metric]}", fontsize=10)

                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
                if not first_png:
                    fig.savefig(Path(out_dir) / "appendix" / f"{name}.png", bbox_inches="tight", pad_inches=0.05)
                    first_png = True
                plt.close(fig)

        add_manifest(manifest, name, kc_csv, "ok")

    except Exception as e:
        add_manifest(manifest, name, kc_csv, "failed", repr(e))


def app_a4_per_config(per_csv, out_dir, manifest):
    name = "app_a4_per_config_specificity_pub_v5"
    try:
        df = read_csv(per_csv)
        mean, _ = metric_mean_std(df, "per_config_asr")

        dcol = id_col(df, "dataset", required=False)
        mcol = id_col(df, "model", required=False)
        kcol = id_col(df, "kc", required=False)
        ccol = id_col(df, "config_id", required=False)
        tcol = id_col(df, "target_label", required=False)
        pcol = id_col(df, "positions", required=False)

        group_cols = [c for c in [dcol, mcol, kcol] if c is not None]
        if not group_cols:
            df["_group"] = "all"
            group_cols = ["_group"]

        out_pdf = Path(out_dir) / "appendix" / f"{name}.pdf"
        ensure_dir(out_pdf.parent)

        first_png = False
        pages = 0

        with PdfPages(out_pdf) as pdf:
            for key, g in df.groupby(group_cols, dropna=False):
                vals = mean.loc[g.index].values

                labels = [f"c{i+1}" for i in range(len(g))]

                fig_w = max(6.5, min(14.0, 0.55 * len(g) + 2))
                fig, ax = plt.subplots(figsize=(fig_w, 3.2), constrained_layout=True)

                x = np.arange(len(g))
                ax.bar(x, percent(vals), width=0.62, color="#A1D99B", edgecolor="black", linewidth=0.55)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=0, ha="center")
                axis_percent(ax, "per_config_asr", vals)
                clean_axis(ax)

                title = " / ".join(map(str, key if isinstance(key, tuple) else [key]))
                ax.set_title(title)

                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
                if not first_png:
                    fig.savefig(Path(out_dir) / "appendix" / f"{name}.png", bbox_inches="tight", pad_inches=0.05)
                    first_png = True
                plt.close(fig)
                pages += 1

        add_manifest(manifest, name, per_csv, "ok", f"pages={pages}")

    except Exception as e:
        add_manifest(manifest, name, per_csv, "failed", repr(e))


def defense_detail(csv_path, out_dir, manifest, name, metrics):
    try:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            add_manifest(manifest, name, csv_path, "skipped", "file missing")
            return

        df = filter_full(read_csv(csv_path))

        available = [m for m in metrics if metric_mean_std(df, m, required=False)[0] is not None]
        if not available:
            add_manifest(manifest, name, csv_path, "skipped", "no metrics")
            return

        fig, axes = plt.subplots(1, len(available), figsize=(3.8 * len(available), 3.0), constrained_layout=True)
        if len(available) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available):
            rows = agg_by_pr(df, metric)
            color = "#FDD0A2" if metric in ["delta_csg", "delta_valid_asr"] else "#9ECAE1"
            chance = 0.5 if "auc" in metric else None
            bar_poison(ax, rows, metric, METRIC_LABEL.get(metric, metric), color=color, chance_line=chance)

        savefig(fig, Path(out_dir) / "appendix" / f"{name}.png")
        add_manifest(manifest, name, csv_path, "ok", f"metrics={available}")

    except Exception as e:
        add_manifest(manifest, name, csv_path, "failed", repr(e))


def app_a5_defense(table_dir, out_dir, manifest):
    table_dir = Path(table_dir)

    defense_detail(
        table_dir / "defense_seed012_strip_summary.csv",
        out_dir, manifest,
        "app_a5_strip_detail_pub_v5",
        ["strip_auc"],
    )
    defense_detail(
        table_dir / "defense_seed012_spectral_signatures_summary.csv",
        out_dir, manifest,
        "app_a5_spectral_signatures_detail_pub_v5",
        ["ss_auc"],
    )
    defense_detail(
        table_dir / "defense_seed012_neural_cleanse_summary.csv",
        out_dir, manifest,
        "app_a5_neural_cleanse_detail_pub_v5",
        ["nc_detects_true_target"],
    )
    defense_detail(
        table_dir / "defense_seed012_fine_pruning_summary.csv",
        out_dir, manifest,
        "app_a5_fine_pruning_detail_pub_v5",
        ["base_valid_asr", "fp_valid_asr", "delta_valid_asr", "base_csg", "fp_csg", "delta_csg"],
    )


def contact_sheet(out_dir, manifest):
    try:
        from PIL import Image, ImageDraw, ImageFont

        out_dir = Path(out_dir)
        pngs = sorted((out_dir / "main").glob("*.png")) + sorted((out_dir / "appendix").glob("*.png"))
        if not pngs:
            return

        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        f = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()

        thumb_w = 620
        margin = 28
        label_h = 32
        cols = 2

        thumbs = []
        for p in pngs:
            im = Image.open(p).convert("RGB")
            h = int(im.size[1] * thumb_w / im.size[0])
            im = im.resize((thumb_w, h), Image.Resampling.LANCZOS)
            thumbs.append((p, im))

        cell_h = max(im.size[1] for _, im in thumbs) + label_h
        rows = math.ceil(len(thumbs) / cols)
        W = cols * thumb_w + (cols + 1) * margin
        H = rows * cell_h + (rows + 1) * margin

        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)

        for i, (p, im) in enumerate(thumbs):
            r = i // cols
            c = i % cols
            x = margin + c * (thumb_w + margin)
            y = margin + r * (cell_h + margin)

            draw.text((x, y), p.name[:75], fill=(0, 0, 0), font=f)
            canvas.paste(im, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + im.size[0], y + label_h + im.size[1]), outline=(210, 210, 210), width=1)

        canvas.save(out_dir / "contact_sheet_pub_v5.png")
        add_manifest(manifest, "contact_sheet_pub_v5", out_dir, "ok", f"png_count={len(pngs)}")

    except Exception as e:
        add_manifest(manifest, "contact_sheet_pub_v5", out_dir, "failed", repr(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", default="results/tables")
    parser.add_argument("--out-dir", default="results/figures/experimental_pub_v5")
    args = parser.parse_args()

    table_dir = Path(args.table_dir)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir / "main")
    ensure_dir(out_dir / "appendix")

    manifest = []

    main_csv = table_dir / "all_dynamic_main_mean_std.csv"
    kc_csv = table_dir / "kc_dynamic_diverse_scalability_mean_std.csv"
    per_csv = table_dir / "per_config_target_specificity_with_positions_mean_std.csv"

    defense_candidates = [
        table_dir / "defense_seed012_paper_compact_mean_std.csv",
        table_dir / "defense_seed012_full_aggregate_by_poison.csv",
        table_dir / "defense_seed012_mean_std.csv",
    ]
    defense_csv = next((p for p in defense_candidates if p.exists()), defense_candidates[0])

    fig5_main_ablation(main_csv, out_dir, manifest)
    fig6_kc(kc_csv, out_dir, manifest)
    fig7_defense(defense_csv, out_dir, manifest)

    app_a1_full_poison(main_csv, out_dir, manifest)
    app_a2_per_setting(main_csv, out_dir, manifest)
    app_a3_kc(kc_csv, out_dir, manifest)
    app_a4_per_config(per_csv, out_dir, manifest)
    app_a5_defense(table_dir, out_dir, manifest)

    contact_sheet(out_dir, manifest)

    man = pd.DataFrame(manifest)
    man_path = out_dir / "figure_generation_manifest.csv"
    man.to_csv(man_path, index=False)

    print("=" * 100)
    print(f"[DONE] output: {out_dir}")
    print(f"[DONE] manifest: {man_path}")
    print("=" * 100)
    print(man.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
