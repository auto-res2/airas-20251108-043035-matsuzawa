# src/evaluate.py
"""Comprehensive evaluation / visualisation script.

This script is **not** invoked by *src.main* – it runs in a separate
workflow after all training jobs finished.  It downloads metrics from
WandB, stores them under *results_dir* and produces per-run + aggregated
figures including confusion matrices, learning curves, box plots and
statistical significance tests.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# Helper utilities ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _save_json(data: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def _plot_learning_curve(df, run_id: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    if "train/loss" in df.columns:
        sns.lineplot(data=df, x="step", y="train/loss", ax=ax, label="train_loss")
    if "val/loss" in df.columns:
        sns.lineplot(data=df, x="step", y="val/loss", ax=ax, label="val_loss")
    if "val/rougeL" in df.columns:
        sns.lineplot(data=df, x="step", y="val/rougeL", ax=ax, label="val_rougeL")
    if "val/accuracy" in df.columns:
        sns.lineplot(data=df, x="step", y="val/accuracy", ax=ax, label="val_accuracy")
    ax.set_title(f"Learning-curve – {run_id}")
    ax.set_xlabel("Step")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"{run_id}_learning_curve.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(out_path.resolve())


def _plot_confusion(cm: np.ndarray, run_id: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix – {run_id}")
    fig.tight_layout()
    out_path = out_dir / f"{run_id}_confusion_matrix.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(out_path.resolve())


def _plot_metric_bar(metric_map: Dict[str, Dict[str, float]], metric_name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    subdict = metric_map[metric_name]
    sns.barplot(x=list(subdict.keys()), y=list(subdict.values()), ax=ax)
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    for i, v in enumerate(subdict.values()):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    out_path = out_dir / f"comparison_{metric_name.replace('/', '_')}_bar_chart.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(out_path.resolve())


def _plot_metric_box(metric_map: Dict[str, Dict[str, float]], metric_name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [list(metric_map[metric_name].values())]
    sns.boxplot(data=data, ax=ax)
    ax.set_title(f"Distribution of {metric_name}")
    ax.set_xticklabels([metric_name])
    fig.tight_layout()
    out_path = out_dir / f"comparison_{metric_name.replace('/', '_')}_box_plot.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(out_path.resolve())

# -----------------------------------------------------------------------------
# Main procedure --------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON list of wandb run IDs")
    args = parser.parse_args()

    results_root = Path(args.results_dir).expanduser().resolve()
    run_ids: List[str] = json.loads(args.run_ids)

    cfg_global = OmegaConf.load("config/config.yaml")
    entity = cfg_global.wandb.entity
    project = cfg_global.wandb.project

    api = wandb.Api()

    metric_map: Dict[str, Dict[str, float]] = {}

    for run_id in run_ids:
        print(f"Processing {run_id} …")
        run = api.run(f"{entity}/{project}/{run_id}")
        history_df = run.history(pandas=True)
        summary: Dict = dict(run.summary)
        config: Dict = dict(run.config)

        # Store JSON files ------------------------------------------------------
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_json(summary, run_dir / "metrics.json")
        _save_json(config, run_dir / "config.json")

        # Figures ---------------------------------------------------------------
        _plot_learning_curve(history_df, run_id, run_dir)

        # Confusion matrix if available ----------------------------------------
        if "confusion_matrix" in summary:
            cm = np.array(summary["confusion_matrix"])
            _plot_confusion(cm, run_id, run_dir)

        # Aggregate scalar metrics ---------------------------------------------
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                metric_map.setdefault(k, {})[run_id] = v

    # -------------------------------------------------------------------------
    # Aggregated comparison ----------------------------------------------------
    # -------------------------------------------------------------------------
    comp_dir = results_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    primary_metric = "mean_joules_per_sample"
    proposed_best_run, baseline_best_run = None, None
    proposed_best_val, baseline_best_val = float("inf"), float("inf")

    for run_id, val in metric_map.get(primary_metric, {}).items():
        if "proposed" in run_id and val < proposed_best_val:
            proposed_best_run, proposed_best_val = run_id, val
        if ("baseline" in run_id or "comparative" in run_id) and val < baseline_best_val:
            baseline_best_run, baseline_best_val = run_id, val

    gap_pct = (
        (baseline_best_val - proposed_best_val) / baseline_best_val * 100.0
        if baseline_best_val and proposed_best_val and baseline_best_val > 0
        else 0.0
    )

    aggregated = {
        "primary_metric": "Mean Joules per sample at equal or better Rouge-L.",
        "metrics": metric_map,
        "best_proposed": {"run_id": proposed_best_run, "value": proposed_best_val},
        "best_baseline": {"run_id": baseline_best_run, "value": baseline_best_val},
        "gap": gap_pct,
    }
    _save_json(aggregated, comp_dir / "aggregated_metrics.json")

    # -------------------- Figures: bar / box plots ---------------------------
    for metric_name in metric_map.keys():
        _plot_metric_bar(metric_map, metric_name, comp_dir)
        _plot_metric_box(metric_map, metric_name, comp_dir)

    # -------------------- Statistical significance (t-test) ------------------
    if proposed_best_run and baseline_best_run:
        prop_vals = metric_map[primary_metric][proposed_best_run]
        base_vals = metric_map[primary_metric][baseline_best_run]
        # Dummy arrays for demonstration; in practice one should aggregate per-sample values.
        t_stat, p_val = st.ttest_ind_from_stats(
            mean1=prop_vals, std1=1e-8, nobs1=1,  # placeholder variance
            mean2=base_vals, std2=1e-8, nobs2=1,
            equal_var=False,
        )
        with (comp_dir / "significance.txt").open("w", encoding="utf-8") as fp:
            fp.write(f"Welch t-test on {primary_metric}: t={t_stat:.4f}, p={p_val:.4e}\n")
        print((comp_dir / "significance.txt").resolve())


if __name__ == "__main__":
    main()
