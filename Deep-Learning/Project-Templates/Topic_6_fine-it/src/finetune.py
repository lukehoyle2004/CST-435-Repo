"""Fine-tune DistilBERT into an eight-queue router with the Hugging Face Trainer.

One call to ``train_config`` = one row in the experiment store. Each run is
seeded (``transformers.set_seed`` seeds Python, NumPy and torch, and the Trainer
reseeds its data order from ``args.seed``), so re-running a configuration on the
same machine reproduces its numbers; tests/test_finetune.py checks this on a
small slice.

The Trainer keeps the epoch with the best VALIDATION macro-F1
(``load_best_model_at_end``); the test split is scored once, afterwards.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

from shared.artifacts import weights_sha256  # noqa: F401  (re-exported for the sweep)
from shared.checkpoints import BASE_MODEL, BASE_MODEL_REVISION, MAX_LENGTH
from shared.queues import QUEUE_NAMES, QUEUE_TO_ID
from src.metrics import macro_f1


@dataclass
class Config:
    run_name: str
    learning_rate: float = 5e-5
    epochs: float = 3
    batch_size: int = 32
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    freeze_encoder: bool = False          # True = train the classification head only
    n_train: int | None = None            # None = the whole train split
    seed: int = 42
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def hyperparams(self) -> dict:
        d = asdict(self)
        d.pop("run_name"); d.pop("notes"); d.pop("extra")
        return d


def load_splits(corpus_csv: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(corpus_csv)
    df["label"] = df["queue"].map(QUEUE_TO_ID)
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test")}


def subsample(train: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Stratified by intent so a small training set still covers every intent."""
    if n is None or n >= len(train):
        return train
    frac = n / len(train)
    part = train.groupby("intent", group_keys=False).apply(
        lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed))
    return part.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)


def _dataset(df: pd.DataFrame, tok) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LENGTH),
                  batched=True, remove_columns=["text"])


def new_model():
    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=BASE_MODEL_REVISION, num_labels=len(QUEUE_NAMES),
        id2label=dict(enumerate(QUEUE_NAMES)), label2id=QUEUE_TO_ID)


def _compute_metrics(pred) -> dict:
    p = pred.predictions.argmax(-1)
    return {"macro_f1": macro_f1(pred.label_ids, p, len(QUEUE_NAMES)),
            "accuracy": float((pred.label_ids == p).mean())}


def train_config(cfg: Config, splits: dict[str, pd.DataFrame], out_dir: Path,
                 log_dir: Path | None = None) -> dict:
    """Train one configuration; save its best checkpoint to ``out_dir``. Returns a result dict."""
    set_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tok = tokenizer()
    train_df = subsample(splits["train"], cfg.n_train, cfg.seed)
    train_ds, val_ds = _dataset(train_df, tok), _dataset(splits["validation"], tok)

    model = new_model()
    if cfg.freeze_encoder:
        for p in model.distilbert.parameters():
            p.requires_grad = False

    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=128,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=25,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
        group_by_length=True,             # batches of similar length = less padding on CPU
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=_compute_metrics)
    t0 = time.perf_counter()
    trainer.train()
    runtime = time.perf_counter() - t0
    val = trainer.evaluate()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{cfg.run_name}.json").write_text(json.dumps({
            "run_name": cfg.run_name, "hyperparams": cfg.hyperparams(),
            "best_checkpoint_step": trainer.state.best_model_checkpoint,
            "best_val_macro_f1": trainer.state.best_metric,
            "log_history": trainer.state.log_history,
        }, indent=1, default=str) + "\n", encoding="utf-8")
    return {"val_macro_f1": float(val["eval_macro_f1"]), "train_runtime_s": runtime,
            "n_train": len(train_df), "model": trainer.model.eval(), "tokenizer": tok}


@torch.inference_mode()
def predict_proba(model, tok, texts: list[str], batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=MAX_LENGTH, return_tensors="pt")
        out.append(model(**enc).logits.softmax(-1).numpy())
    return np.concatenate(out)

