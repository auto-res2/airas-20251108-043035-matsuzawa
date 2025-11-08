# src/main.py
"""Hydra launcher that spawns *src.train* as a subprocess.

Example (full):
    uv run python -u -m src.main run=proposed-iter1 results_dir=./results mode=full
Example (trial):
    uv run python -u -m src.main run=proposed-iter1 results_dir=./results mode=trial
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_overrides(cfg: DictConfig) -> List[str]:
    return [
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]


@hydra.main(config_path="../config")
def main(cfg: DictConfig):  # noqa: D401
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    # Persist composed (yet minimal) config – the *train* subprocess will merge run-specific YAML.
    run_dir = Path(cfg.results_dir).expanduser() / str(cfg.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "orchestrator_config.yaml")

    python = sys.executable
    cmd = [python, "-u", "-m", "src.train"] + _build_overrides(cfg)
    print("Launching subprocess:", " ".join(cmd))
    env = os.environ.copy()
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    main()
