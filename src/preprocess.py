# src/preprocess.py
"""Data-loading & tokenisation utilities.

Supports two task families:
1. *Summarisation* (CNN/DailyMail etc.) – causal-LM style training with
   labels equal to target summaries.
2. *Classification* (GLUE-like) – standard text classification.
"""
from __future__ import annotations

import functools
from typing import Tuple

from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

CACHE_DIR = ".cache/"

# -----------------------------------------------------------------------------
# Tokeniser helpers ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _mask_pad(ids: list[int], pad_id: int):
    return [-100 if tok == pad_id else tok for tok in ids]


def _tok_summarisation(examples, *, tokenizer: AutoTokenizer, cfg: DictConfig):
    inputs = tokenizer(
        examples[cfg.dataset.text_column],
        max_length=cfg.dataset.max_length,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            examples[cfg.dataset.summary_column],
            max_length=cfg.evaluation.generate_max_length,
            truncation=True,
            padding="max_length",
        )
    pad_id = tokenizer.pad_token_id
    inputs["labels"] = [_mask_pad(seq, pad_id) for seq in targets["input_ids"]]
    return inputs


def _tok_classification(examples, *, tokenizer: AutoTokenizer, cfg: DictConfig):
    out = tokenizer(
        examples[cfg.dataset.text_column],
        max_length=cfg.dataset.max_length,
        truncation=True,
        padding="max_length",
    )
    out["labels"] = examples[cfg.dataset.label_column]
    return out

# -----------------------------------------------------------------------------
# Public API ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_dataloaders(
    cfg: DictConfig, tokenizer: AutoTokenizer
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    raw = load_dataset(cfg.dataset.name, cfg.dataset.get("subset", None), cache_dir=CACHE_DIR)

    if cfg.dataset.task == "summarisation":
        map_fn = functools.partial(_tok_summarisation, tokenizer=tokenizer, cfg=cfg)
    else:
        map_fn = functools.partial(_tok_classification, tokenizer=tokenizer, cfg=cfg)

    proc = raw.map(
        map_fn,
        batched=True,
        num_proc=cfg.dataset.num_workers,
        remove_columns=raw[cfg.dataset.train_split].column_names,
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if cfg.training.precision == "fp16" else None,
    )

    def _mk_loader(split: str, shuffle: bool, batch_size: int):
        return DataLoader(
            proc[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.training.pin_memory,
            persistent_workers=cfg.training.persistent_workers,
            collate_fn=collator,
        )

    train_dl = _mk_loader(cfg.dataset.train_split, True, cfg.dataset.batch_size)
    val_dl = _mk_loader(cfg.dataset.validation_split, False, cfg.evaluation.eval_batch_size)
    test_split = cfg.dataset.get("test_split", cfg.dataset.validation_split)
    test_dl = _mk_loader(test_split, False, cfg.evaluation.eval_batch_size)
    return train_dl, val_dl, test_dl
