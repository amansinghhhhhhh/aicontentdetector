"""Quick CPU fine-tune of RoBERTa-base on HC3 detection data.

Usage:
    python -m ml.train --epochs 1 --batch 4 --accum 2
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ml import config as cfg

BASE_MODEL = "FacebookAI/roberta-base"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EncodedDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int, limit: int = 0):
        self.input_ids: List[List[int]] = []
        self.attention: List[List[int]] = []
        self.labels: List[int] = []

        with open(path, encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if limit and i >= limit:
                    break
                row = json.loads(line)
                enc = tokenizer(
                    row["text"],
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                )
                self.input_ids.append(enc["input_ids"])
                self.attention.append(enc["attention_mask"])
                self.labels.append(int(row["label"]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.attention[idx], self.labels[idx]


def collate(batch):
    ids = [b[0] for b in batch]
    mask = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    pad_id = collate.tokenizer.pad_token_id
    max_len = max(len(x) for x in ids)
    padded_ids = torch.full((len(ids), max_len), pad_id, dtype=torch.long)
    padded_mask = torch.zeros((len(ids), max_len), dtype=torch.long)
    for i, (seq, m) in enumerate(zip(ids, mask)):
        padded_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        padded_mask[i, : len(seq)] = torch.tensor(m, dtype=torch.long)
    return padded_ids, padded_mask, labels


@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    scores: List[float] = []
    labels: List[int] = []
    total_loss, n_batches = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss()
    for ids, mask, y in loader:
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1
        probs = torch.softmax(logits.float(), dim=-1)[:, 1]
        scores.extend(probs.tolist())
        labels.extend(y.tolist())
    model.train()
    return {
        "loss": total_loss / max(1, n_batches),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, [1 if s >= 0.5 else 0 for s in scores])),
        "scores": scores,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default=str(cfg.ROOT / "data" / "hc3" / "train.jsonl"))
    parser.add_argument("--val-file", default=str(cfg.ROOT / "data" / "hc3" / "val.jsonl"))
    parser.add_argument("--out", default=str(cfg.ROOT / "models" / "ft-hc3-roberta"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, default=0, help="cap training rows (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads())))

    print(f"loading tokenizer/model: {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    collate.tokenizer = tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.train()

    train_set = EncodedDataset(Path(args.train_file), tokenizer, args.max_len, args.limit)
    val_set = EncodedDataset(Path(args.val_file), tokenizer, args.max_len)
    print(f"train={len(train_set)} val={len(val_set)}", flush=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch * 2, shuffle=False, collate_fn=collate
    )

    steps_per_epoch = max(1, len(train_loader) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.06)),
        num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss()
    grad_clip = 1.0

    best_auc = -1.0
    best_metrics = {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    running_loss, micro_steps = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for step, (ids, mask, y) in enumerate(train_loader, start=1):
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(logits, y) / args.accum
            loss.backward()
            running_loss += loss.item() * args.accum
            micro_steps += 1

            if step % args.accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if micro_steps % 25 == 0:
                    elapsed = time.perf_counter() - started
                    done = micro_steps / len(train_loader)
                    eta = elapsed / max(done, 1e-6) - elapsed
                    print(
                        f"epoch={epoch} micro={micro_steps}/{len(train_loader)} "
                        f"loss={running_loss / micro_steps:.4f} "
                        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                        flush=True,
                    )

        metrics = evaluate(model, val_loader)
        metrics.pop("scores")
        metrics.pop("labels")
        history.append({"epoch": epoch, **metrics})
        print(
            f"[epoch {epoch}] val_loss={metrics['loss']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} acc={metrics['accuracy']:.4f}",
            flush=True,
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_metrics = dict(metrics)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"saved best checkpoint -> {out_dir} (auc={best_auc:.4f})", flush=True)

    result = {
        "best": {k: v for k, v in best_metrics.items() if k != "scores"},
        "history": history,
        "args": vars(args),
        "train_rows": len(train_set),
        "val_rows": len(val_set),
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"DONE best_val_auc={best_auc:.4f} total={result['minutes']}m", flush=True)


if __name__ == "__main__":
    main()
