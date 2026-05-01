"""
09_train_slm.py
Train DeBERTa-v3-base for binary fake-news classification.

Label: 0=True, 1=Fake

Reads slm_data/{method}/{train,val,test}.jsonl
Writes to slm_outputs/{method}/:
  best_model/              - best checkpoint by macro_f1
  test_predictions.jsonl   - per-sample predictions
  metrics.json             - accuracy, macro_f1, f1_fake, f1_true, json_parse_rate

Usage:
  python scripts/09_train_slm.py --method basepack_only
  python scripts/09_train_slm.py --method basepack_only --force
  python scripts/09_train_slm.py --method heuristic_pre
  python scripts/09_train_slm.py \
      --train slm_data/custom/train.jsonl \
      --val   slm_data/custom/val.jsonl \
      --test  slm_data/custom/test.jsonl \
      --output_dir slm_outputs/custom
"""

import json
import argparse
import os
from pathlib import Path

# RTX 4000 系列不支持 P2P/IB 通信，训练前显式禁用
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score


def slm_outputs_complete(output_dir: Path) -> bool:
    """
    判断该 output_dir 是否已有完整训练产物（metrics、测试预测、best_model 权重）。
    用于再次跑 pipeline 时跳过已完成的 SLM 训练。
    """
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    preds_path = output_dir / "test_predictions.jsonl"
    best = output_dir / "best_model"
    if not metrics_path.is_file() or not preds_path.is_file() or not best.is_dir():
        return False
    has_weights = (best / "model.safetensors").is_file() or (best / "pytorch_model.bin").is_file()
    has_config = (best / "config.json").is_file()
    return bool(has_weights and has_config)


class TextDataset(Dataset):
    def __init__(self, records: list, tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(rec["label"]), dtype=torch.long),
        }


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    f1_fake = f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
    f1_true = f1_score(labels, preds, pos_label=0, average="binary", zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_fake": f1_fake,
        "f1_true": f1_true,
    }


def train_slm(train_path: Path, val_path: Path, test_path: Path,
              output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    print(f"Model:       {args.model_name}")
    print(f"Output dir:  {output_dir}")
    print(f"Train:       {train_path}")
    print(f"Val:         {val_path}")
    print(f"Test:        {test_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    test_records = load_jsonl(test_path)

    print(f"  train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    train_ds = TextDataset(train_records, tokenizer, args.max_length)
    val_ds = TextDataset(val_records, tokenizer, args.max_length)
    test_ds = TextDataset(test_records, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        dataloader_num_workers=4,
        logging_steps=20,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"Best model saved to {best_model_dir}")

    print("Running test evaluation...")
    test_output = trainer.predict(test_ds)
    logits = test_output.predictions
    preds = np.argmax(logits, axis=-1)
    labels = [int(r["label"]) for r in test_records]

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    f1_fake = f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
    f1_true = f1_score(labels, preds, pos_label=0, average="binary", zero_division=0)

    metrics = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "f1_fake": round(float(f1_fake), 4),
        "f1_true": round(float(f1_true), 4),
        "num_test": len(test_records),
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"  F1-Fake:   {f1_fake:.4f}")
    print(f"  F1-True:   {f1_true:.4f}")

    preds_path = output_dir / "test_predictions.jsonl"
    with open(preds_path, "w") as f:
        for rec, pred in zip(test_records, preds):
            out = {
                "event_id": rec.get("event_id", ""),
                "gold_label": int(rec["label"]),
                "pred_label": int(pred),
                "correct": int(pred) == int(rec["label"]),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Predictions saved: {preds_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa SLM classifier")
    parser.add_argument("--method", default=None,
                        choices=["basepack_only", "heuristic_pre", "sft_hint_pre", "dpo_hint_pre"],
                        help="If set, auto-sets train/val/test/output_dir paths")
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 slm_outputs 下已有完整输出也重新训练",
    )
    args = parser.parse_args()

    if args.method:
        args.train = f"slm_data/{args.method}/train.jsonl"
        args.val = f"slm_data/{args.method}/val.jsonl"
        args.test = f"slm_data/{args.method}/test.jsonl"
        args.output_dir = f"slm_outputs/{args.method}"

    if not all([args.train, args.val, args.test, args.output_dir]):
        raise ValueError("Provide --method OR all of --train --val --test --output_dir")

    out_path = Path(args.output_dir)
    if not args.force and slm_outputs_complete(out_path):
        print(f"跳过 SLM 训练（已有完整输出）: {out_path}")
        print("  若要强制重训，请加参数: --force")
        return

    train_slm(
        Path(args.train), Path(args.val), Path(args.test),
        Path(args.output_dir), args,
    )


if __name__ == "__main__":
    main()
