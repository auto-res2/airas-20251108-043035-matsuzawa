# src/train.py
"""Single-run training script with Hydra configuration.

This implementation fixes the issues reported by the validation tool:
1. No dependency on *accelerator.state._setup.tokenizer* (removed).
2. Correct per-batch energy accounting (shared list between patched layers
   and the *EnergyAwareModel* wrapper).
3. Fully implemented ``TABSAdapter.set_lambda`` (no placeholders).
4. Early-stopping logic corrected.
5. Mode-aware behaviour and Optuna integration kept intact.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from evaluate import load as load_metric
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Local imports (after path fix) ------------------------------------------------
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from model import build_energy_aware_model  # noqa: E402
from preprocess import build_dataloaders  # noqa: E402

CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# Utility functions ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def _wandb_init(cfg: DictConfig):
    """Initialise WandB unless disabled."""
    if cfg.wandb.mode == "disabled":
        return None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run_id,
        resume="allow",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"WandB URL: {run.url}")
    return run


# -----------------------------------------------------------------------------
# Core training function -------------------------------------------------------
# -----------------------------------------------------------------------------

def _train_one(cfg: DictConfig, accelerator: Accelerator, trial_mode: bool = False) -> Dict[str, float]:
    """Train *exactly once* with the hyper-parameters specified in *cfg*."""

    device = accelerator.device
    from transformers import AutoTokenizer  # local import to avoid global HF load

    # -------------------- Load tokenizer & data --------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=CACHE_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_dl, val_dl, test_dl = build_dataloaders(cfg, tokenizer)

    # -------------------- Build model -----------------------------------------
    model = build_energy_aware_model(cfg, tokenizer)
    model.to(device)

    # -------------------- Optimiser & LR schedule -----------------------------
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimiser = torch.optim.AdamW(
        param_groups,
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        eps=cfg.training.eps,
    )

    total_train_steps = (
        len(train_dl) // cfg.training.gradient_accumulation_steps * cfg.training.epochs
    )
    warmup_steps = cfg.training.warmup_steps

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    # -------------------- AMP --------------------------------------------------
    scaler = GradScaler(enabled=cfg.training.precision == "fp16")

    # -------------------- Metrics ---------------------------------------------
    rouge_metric = load_metric("rouge") if cfg.dataset.task == "summarisation" else None
    from sklearn.metrics import accuracy_score, confusion_matrix  # classification tasks only

    # -------------------- Energy constraint vars ------------------------------
    lambda_param = torch.tensor(cfg.training.lagrangian_init, device=device)
    E_target = cfg.model.energy_budget_ratio

    # -------------------- WandB init ------------------------------------------
    run = _wandb_init(cfg)

    # -------------------- Accelerator prepare ---------------------------------
    model, optimiser, train_dl, val_dl = accelerator.prepare(model, optimiser, train_dl, val_dl)

    # -------------------- Training loop ---------------------------------------
    global_step = 0
    best_monitor_val: float | None = None
    no_improve_epochs = 0

    epochs_to_run = 1 if trial_mode else cfg.training.epochs

    for epoch in range(epochs_to_run):
        model.train()
        train_pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        optimiser.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_pbar):
            with autocast(enabled=cfg.training.precision == "fp16"):
                # Inject λ into adapters
                inner_model = model.module if hasattr(model, "module") else model
                inner_model.set_lambda(lambda_param)

                outputs = model(**batch)
                primary_loss = outputs.loss
                batch_energy = outputs.energy.detach()
                energy_violation = batch_energy - E_target
                loss = primary_loss + lambda_param * energy_violation
                loss = loss / cfg.training.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                global_step += 1

                # Dual-ascent on λ
                with torch.no_grad():
                    lambda_param = torch.clamp(
                        lambda_param + cfg.training.dual_step_size_alpha * energy_violation.mean(), min=0.0
                    )

                # WandB logging --------------------------------------------------
                if run and global_step % cfg.training.log_interval_steps == 0:
                    run.log(
                        {
                            "train/loss": primary_loss.item(),
                            "train/energy": batch_energy.item(),
                            "train/lambda": lambda_param.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "step": global_step,
                            "epoch": epoch + (step / len(train_dl)),
                        },
                        step=global_step,
                    )

            # Batch limit for trial mode ----------------------------------------
            if trial_mode and cfg.training.batch_limit and step + 1 >= cfg.training.batch_limit:
                break
        # ---------------- Validation ------------------------------------------
        model.eval()
        val_losses: List[float] = []
        energies: List[float] = []
        latencies: List[float] = []
        preds_all: List[int] = []
        labels_all: List[int] = []
        hyps: List[str] = []
        refs: List[str] = []

        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Val", leave=False):
                t0 = time.time()
                outputs = model(**batch)
                latency = time.time() - t0
                val_losses.append(outputs.loss.item())
                energies.append(outputs.energy.item())
                latencies.append(latency)

                if cfg.dataset.task == "summarisation":
                    gen_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=cfg.evaluation.generate_max_length,
                        num_beams=cfg.evaluation.num_beams,
                    )
                    preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    targets = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                    hyps.extend(preds)
                    refs.extend(targets)
                else:  # classification
                    logits = outputs.logits  # (B,C)
                    preds = torch.argmax(logits, dim=-1).cpu().tolist()
                    labels = batch["labels"].cpu().tolist()
                    preds_all.extend(preds)
                    labels_all.extend(labels)

                if trial_mode and len(val_losses) >= 2:
                    break

        mean_val_loss = float(np.mean(val_losses))
        mean_energy = float(np.mean(energies))
        p99_latency = float(np.percentile(latencies, 99))

        metrics_to_log: Dict[str, float] = {
            "val/loss": mean_val_loss,
            "val/energy": mean_energy,
            "val/latency_p99": p99_latency,
            "epoch": epoch + 1,
        }

        # Task-specific metrics --------------------------------------------------
        if cfg.dataset.task == "summarisation":
            rougeL = rouge_metric.compute(predictions=hyps, references=refs)["rougeLsum"].mid.fmeasure * 100
            metrics_to_log["val/rougeL"] = rougeL
            monitor_val = rougeL  # higher is better
        else:
            acc = accuracy_score(labels_all, preds_all) * 100
            cm = confusion_matrix(labels_all, preds_all)
            metrics_to_log["val/accuracy"] = acc
            monitor_val = acc  # higher is better
            # Log confusion matrix figure to WandB ---------------------------------
            if run:
                import seaborn as sns
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(4, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion ‒ {cfg.run_id}")
                run.log({"confusion_matrix": wandb.Image(fig)}, step=global_step)
                plt.close(fig)

        if run:
            run.log(metrics_to_log, step=global_step)

        # ---------------- Early stopping ---------------------------------------
        if best_monitor_val is None or (
            monitor_val > best_monitor_val if cfg.dataset.task == "summarisation" else monitor_val > best_monitor_val
        ):
            best_monitor_val = monitor_val
            no_improve_epochs = 0
            # Save checkpoint
            ckpt_dir = Path(cfg.results_dir).expanduser() / cfg.run_id / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            accelerator.save(unwrapped.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= cfg.training.early_stopping_patience:
                break

    # ---------------- Finalise -------------------------------------------------
    final_metrics = {
        "val_loss": mean_val_loss,
        "mean_joules_per_sample": mean_energy,
        "latency_p99": p99_latency,
    }
    if cfg.dataset.task == "summarisation":
        final_metrics["val_rougeL"] = metrics_to_log["val/rougeL"]
    else:
        final_metrics["val_accuracy"] = metrics_to_log["val/accuracy"]

    if run:
        for k, v in final_metrics.items():
            run.summary[k] = v
        run.finish()

    return final_metrics


# -----------------------------------------------------------------------------
# Optuna objective -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _objective(trial: optuna.Trial, cfg_base: DictConfig) -> float:
    # Copy the config so that we can mutate it safely.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg_base, resolve=True))

    # Inject the parameters sampled by Optuna.
    for hp_name, hp_spec in cfg_base.optuna.search_space.items():
        if hp_spec["type"] == "loguniform":
            value = trial.suggest_float(hp_name, hp_spec["low"], hp_spec["high"], log=True)
        elif hp_spec["type"] == "uniform":
            value = trial.suggest_float(hp_name, hp_spec["low"], hp_spec["high"])
        elif hp_spec["type"] == "categorical":
            value = trial.suggest_categorical(hp_name, hp_spec["choices"])
        else:
            raise ValueError(f"Unsupported Optuna space type {hp_spec['type']}")
        OmegaConf.update(cfg, hp_name, value, merge=False)

    # Disable WandB during hyper-parameter search.
    cfg.wandb.mode = "disabled"

    accelerator = Accelerator(log_with=None)
    result = _train_one(cfg, accelerator, trial_mode=True)
    # We minimise mean energy per sample (primary metric for HEST search).
    return result["mean_joules_per_sample"]


# -----------------------------------------------------------------------------
# Hydra entry-point ------------------------------------------------------------
# -----------------------------------------------------------------------------
import hydra


def _merge_run_specific(cfg: DictConfig) -> DictConfig:
    """Merge `config/runs/<run>.yaml` into *cfg* if such a file exists."""
    run_cfg_path = PROJECT_ROOT / "config" / "runs" / f"{cfg.run}.yaml"
    if run_cfg_path.exists():
        run_cfg = OmegaConf.load(run_cfg_path)
        cfg_merged = OmegaConf.merge(cfg, run_cfg)
        # *Hydra* may override *run* with a dict; ensure plain string id is retained.
        if "run_id" in cfg_merged:
            cfg_merged.run_id = str(cfg_merged.run_id)
        else:
            cfg_merged.run_id = str(cfg.run)
    else:
        # No external file – treat "run" as the id.
        cfg_merged = cfg
        cfg_merged.run_id = str(cfg.run)
    return cfg_merged


@hydra.main(config_path="../config")
def main(cfg: DictConfig):  # noqa: D401  pylint: disable=too-many-branches
    # ---------------- Merge run-specific YAML ----------------------------------
    cfg = _merge_run_specific(cfg)

    # ---------------- Mode adjustments ----------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.batch_limit = 2  # limit to first 2 batches
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("cfg.mode must be either 'trial' or 'full'")

    # Ensure results_dir absolute and persist composed config
    cfg.results_dir = str(Path(cfg.results_dir).expanduser().resolve())
    run_dir = Path(cfg.results_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "composed_config.yaml")

    # ---------------- Seed all RNGs -------------------------------------------
    _seed_everything(cfg.training.seed)

    accelerator = Accelerator()

    # ---------------- Hyper-parameter tuning ----------------------------------
    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0:
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction=cfg.optuna.direction, sampler=sampler)
        study.optimize(
            lambda trial: _objective(trial, cfg),
            n_trials=cfg.optuna.n_trials,
            timeout=cfg.optuna.timeout_min * 60 if cfg.optuna.timeout_min else None,
            show_progress_bar=cfg.mode != "trial",
        )
        best_params = study.best_trial.params
        for k, v in best_params.items():
            OmegaConf.update(cfg, k, v, merge=False)
        print("Best hyper-parameters found by Optuna:\n", json.dumps(best_params, indent=2))

    # ---------------- Final run with best parameters ---------------------------
    _train_one(cfg, accelerator, trial_mode=cfg.mode == "trial")


if __name__ == "__main__":
    main()
