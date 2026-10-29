"""Download HC3 domain jsonl files and build balanced, leakage-free splits.

Each question yields one human answer + one ChatGPT answer (a "group"), so both
splits and deduplication happen at question level — no answer from a held-out
question can leak into training.

Usage:
    python -m ml.prepare_hc3 --groups 2000
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download

from ml import config as cfg

REPO_ID = "Hello-SimpleAI/HC3"
FILES = [
    "finance.jsonl",
    "medicine.jsonl",
    "open_qa.jsonl",
    "reddit_eli5.jsonl",
    "wiki_csai.jsonl",
]
OUT_DIR = cfg.ROOT / "data" / "hc3"
WORD_RE = re.compile(r"\b[\w'-]+\b")


def _words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _acceptable(text: str) -> bool:
    n = _words(text)
    return 8 <= n <= 450


def load_groups() -> List[dict]:
    groups: List[dict] = []
    seen_texts = set()
    per_file: Dict[str, int] = defaultdict(int)

    for filename in FILES:
        path = hf_hub_download(REPO_ID, filename, repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                human_answers = row.get("human_answers") or []
                ai_answers = row.get("chatgpt_answers") or []

                human = next(
                    (a.strip() for a in human_answers if isinstance(a, str) and _acceptable(a.strip())),
                    None,
                )
                ai = next(
                    (a.strip() for a in ai_answers if isinstance(a, str) and _acceptable(a.strip())),
                    None,
                )
                if human is None or ai is None:
                    continue
                if human == ai or human in seen_texts or ai in seen_texts:
                    continue
                seen_texts.update((human, ai))
                groups.append(
                    {
                        "group": f"{Path(filename).stem}:{len(groups)}",
                        "source": Path(filename).stem,
                        "human": human,
                        "ai": ai,
                    }
                )
                per_file[Path(filename).stem] += 1

    print(f"loaded {len(groups)} usable question groups: {dict(per_file)}")
    return groups


def balanced_sample(groups: List[dict], target_groups: int, seed: int) -> List[dict]:
    if target_groups >= len(groups):
        return groups
    rng = random.Random(seed)
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for group in groups:
        by_source[group["source"]].append(group)

    quota = max(1, target_groups // len(by_source))
    sampled: List[dict] = []
    for source, items in by_source.items():
        rng.shuffle(items)
        sampled.extend(items[:quota])
    rng.shuffle(sampled)
    missing = target_groups - len(sampled)
    if missing > 0:
        rest = [g for g in groups if g not in set(sampled)]
        rng.shuffle(rest)
        sampled.extend(rest[:missing])
    return sampled[:target_groups]


def split_groups(groups: List[dict], seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * 0.78)
    n_val = int(n * 0.11)
    return {
        "train": groups[:n_train],
        "val": groups[n_train:n_train + n_val],
        "test": groups[n_train + n_val:],
    }


def write_split(name: str, groups: List[dict]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.jsonl"
    rows = 0
    with open(path, "w", encoding="utf-8") as handle:
        for group in groups:
            for label, text in ((0, group["human"]), (1, group["ai"])):
                handle.write(
                    json.dumps(
                        {"text": text, "label": label, "group": group["group"], "source": group["source"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups = load_groups()
    sampled = balanced_sample(groups, args.groups, args.seed)
    splits = split_groups(sampled, args.seed)

    summary = {}
    for name, items in splits.items():
        rows = write_split(name, items)
        summary[name] = {"groups": len(items), "rows": rows}
    print(f"written to {OUT_DIR}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
