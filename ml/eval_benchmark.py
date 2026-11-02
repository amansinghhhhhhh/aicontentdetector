"""Baseline evaluation of the serving detector and humanizer on the public benchmark.

Scores every English sample in the cached
``wasitaigeneratedcom/ai-detector-benchmark-2026-09`` dataset with the serving
detector, reports threshold/calibration metrics, and measures a humanize
evasion baseline against the local stealth proxies.

Run from the repository root:
    python -m ml.eval_benchmark
    python -m ml.eval_benchmark --limit 60
    python -m ml.eval_benchmark --skip-humanize
    python -m ml.eval_benchmark --parquet path/to/benchmark.parquet
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf-home"))
os.environ.setdefault("DETECTOR_BACKEND", "onnx")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from ml import config as cfg  # noqa: E402
from ml.detect import score_many  # noqa: E402
from ml.humanize import humanize  # noqa: E402

DATASET_GLOB = str(
    ROOT
    / "models"
    / "hf-home"
    / "hub"
    / "datasets--wasitaigeneratedcom--ai-detector-benchmark-2026-09"
    / "snapshots"
    / "*"
    / "data"
    / "benchmark.parquet"
)
RUNS_DIR = ROOT / "ml" / "runs"
SCORE_CSV = RUNS_DIR / "eval_scores.csv"
REPORT_JSON = RUNS_DIR / "eval_baseline.json"
CSV_COLUMNS = ["id", "label", "words", "source_dataset", "generator", "ai_percent", "verdict", "error"]
WORD_BUCKETS = [("short<150", 0, 150), ("medium150-400", 150, 400), ("long>400", 400, 10**9)]
SCORE_CHUNK = 48


def _find_parquet(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Parquet not found: {path}")
        return path
    matches = sorted(glob.glob(DATASET_GLOB))
    if not matches:
        raise SystemExit(
            "Benchmark parquet not found in the HF cache. Pass --parquet PATH "
            "or download wasitaigeneratedcom/ai-detector-benchmark-2026-09."
        )
    return Path(matches[-1])


def _load_dataset(parquet: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    path = _find_parquet(parquet)
    df = pd.read_parquet(path)
    df = df[(df["text_available"] == True) & (df["lang"] == "en")].copy()  # noqa: E712
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    df = df.drop_duplicates(subset=["id"])
    df["id"] = df["id"].astype(str)
    df["words"] = pd.to_numeric(df["words"], errors="coerce").fillna(
        df["text"].astype(str).str.split().str.len()
    )
    if limit and limit < len(df):
        parts = []
        shares = df["label"].value_counts(normalize=True)
        for label, share in shares.items():
            pool = df[df["label"] == label]
            n = max(1, int(round(limit * share)))
            parts.append(pool.sample(n=min(n, len(pool)), random_state=7))
        df = pd.concat(parts).drop_duplicates(subset=["id"]).head(limit).copy()
    print(f"dataset: {path.name}  samples={len(df)}  labels={dict(df['label'].value_counts())}")
    return df.reset_index(drop=True)


def _score_batch(texts: list) -> list:
    try:
        scores = score_many(texts)
        if len(scores) == len(texts):
            return scores
    except Exception:
        pass
    out = []
    for text in texts:
        try:
            res = score_many([text])
            out.append(res[0] if res else None)
        except Exception:
            out.append(None)
    return out


def _score_dataset(df: pd.DataFrame, fresh: bool) -> pd.DataFrame:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame(columns=CSV_COLUMNS)
    if SCORE_CSV.exists() and not fresh:
        try:
            cached = pd.read_csv(SCORE_CSV)
        except Exception:
            cached = pd.DataFrame(columns=CSV_COLUMNS)
    done = set()
    if len(cached):
        ok = cached[cached["ai_percent"].notna()]
        done = set(ok["id"].astype(str).tolist()) if "id" in ok else set()
    todo = df[~df["id"].astype(str).isin(done)].copy()
    print(f"scoring: {len(df) - len(todo)} cached, {len(todo)} to go")

    rows = cached.to_dict("records") if len(cached) else []
    started = time.perf_counter()
    total = len(todo)
    for start in range(0, total, SCORE_CHUNK):
        batch = todo.iloc[start : start + SCORE_CHUNK]
        texts = batch["text"].astype(str).tolist()
        scores = _score_batch(texts)
        for (_, row), score in zip(batch.iterrows(), scores):
            rows.append(
                {
                    "id": str(row["id"]),
                    "label": row["label"],
                    "words": int(row["words"]),
                    "source_dataset": row.get("source_dataset"),
                    "generator": row.get("generator"),
                    "ai_percent": round(float(score["ai_percent"]), 2) if score else None,
                    "verdict": score["verdict"] if score else None,
                    "error": None if score else "scoring failed",
                }
            )
        pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(SCORE_CSV, index=False)
        done_n = min(start + len(batch), total)
        elapsed = time.perf_counter() - started
        rate = done_n / elapsed if elapsed > 0 else 0
        eta = (total - done_n) / rate if rate > 0 else 0
        print(
            f"  [{done_n}/{total}] {elapsed:6.1f}s  {rate:5.1f} texts/s  eta {eta/60:5.1f} min",
            flush=True,
        )

    scored = pd.DataFrame(rows, columns=CSV_COLUMNS)
    scored["id"] = scored["id"].astype(str)
    df["id"] = df["id"].astype(str)
    merged = df.merge(scored[["id", "ai_percent", "verdict", "error"]], on="id", how="left")
    failed = merged["ai_percent"].isna().sum()
    if failed:
        print(f"WARNING: {failed} samples could not be scored (see {SCORE_CSV.name})")
    return merged


def _binary_thresholds(y: np.ndarray, scores: np.ndarray) -> list:
    out = []
    for thr in (25, 40, 50, 60, 70):
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out.append(
            {
                "threshold": thr,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "fpr": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
                "fnr": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
                "accuracy": round((tp + tn) / len(y), 4),
            }
        )
    return out


def _optimal_threshold(y: np.ndarray, scores: np.ndarray) -> dict:
    candidates = np.unique(np.round(scores, 1))
    best_f1 = (0.0, 50.0)
    best_youden = (0.0, 50.0)
    for thr in candidates:
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if f1 > best_f1[0]:
            best_f1 = (float(f1), float(thr))
        if (tpr - fpr) > best_youden[0]:
            best_youden = (float(tpr - fpr), float(thr))
    return {
        "f1_max": {"threshold": best_f1[1], "f1": round(best_f1[0], 4)},
        "youden_max": {"threshold": best_youden[1], "j": round(best_youden[0], 4)},
    }


def _detector_report(df: pd.DataFrame) -> dict:
    valid = df[df["ai_percent"].notna()].copy()
    binary = valid[valid["label"].isin(["human", "ai"])]
    y = (binary["label"] == "ai").astype(int).values
    scores = binary["ai_percent"].values
    report = {
        "n_valid": int(len(valid)),
        "n_binary": int(len(binary)),
        "n_human": int((binary["label"] == "human").sum()),
        "n_ai": int((binary["label"] == "ai").sum()),
        "current_thresholds": {
            "ai_verdict": cfg.AI_VERDICT_THRESHOLD,
            "mixed_verdict": cfg.MIXED_VERDICT_THRESHOLD,
        },
    }
    if len(set(y)) == 2:
        report["auc"] = round(float(roc_auc_score(y, scores)), 4)
        report["average_precision"] = round(float(average_precision_score(y, scores)), 4)
        report["thresholds"] = _binary_thresholds(y, scores)
        report["optimal"] = _optimal_threshold(y, scores)

    label_stats = {}
    for label in ("human", "ai", "humanized", "ai_edited"):
        sub = valid[valid["label"] == label]
        if not len(sub):
            continue
        vals = sub["ai_percent"].values
        label_stats[label] = {
            "n": int(len(sub)),
            "mean_ai_percent": round(float(np.mean(vals)), 2),
            "median_ai_percent": round(float(np.median(vals)), 2),
            "flagged_ai_60": round(float((vals >= 60).mean()), 4),
            "not_human_25": round(float((vals >= 25).mean()), 4),
        }
    report["by_label"] = label_stats

    bucket_stats = {}
    for name, low, high in WORD_BUCKETS:
        sub = valid[(valid["words"] >= low) & (valid["words"] < high)]
        if not len(sub):
            continue
        entry = {"n": int(len(sub))}
        h = sub[sub["label"] == "human"]["ai_percent"].values
        a = sub[sub["label"] == "ai"]["ai_percent"].values
        if len(h):
            entry["human_mean"] = round(float(np.mean(h)), 2)
            entry["human_flagged_60"] = round(float((h >= 60).mean()), 4)
        if len(a):
            entry["ai_mean"] = round(float(np.mean(a)), 2)
            entry["ai_recall_60"] = round(float((a >= 60).mean()), 4)
        if len(h) and len(a):
            entry["auc"] = round(
                float(roc_auc_score([0] * len(h) + [1] * len(a), np.concatenate([h, a]))), 4
            )
        bucket_stats[name] = entry
    report["by_words"] = bucket_stats

    src_stats = {}
    top_sources = valid["source_dataset"].value_counts().head(8)
    for source in top_sources.index:
        sub = valid[valid["source_dataset"] == source]
        h = sub[sub["label"] == "human"]["ai_percent"].values
        a = sub[sub["label"] == "ai"]["ai_percent"].values
        entry = {"n": int(len(sub))}
        if len(h):
            entry["human_mean"] = round(float(np.mean(h)), 2)
        if len(a):
            entry["ai_mean"] = round(float(np.mean(a)), 2)
        if len(h) and len(a):
            entry["auc"] = round(
                float(roc_auc_score([0] * len(h) + [1] * len(a), np.concatenate([h, a]))), 4
            )
        src_stats[str(source)] = entry
    report["by_source"] = src_stats

    gen_stats = {}
    ai_only = valid[valid["label"] == "ai"]
    for gen in ai_only["generator"].value_counts().head(8).index:
        sub = ai_only[ai_only["generator"] == gen]
        values = sub["ai_percent"].values
        gen_stats[str(gen)] = {
            "n": int(len(sub)),
            "mean_ai_percent": round(float(np.mean(values)), 2),
            "recall_60": round(float((values >= 60).mean()), 4),
        }
    report["by_generator"] = gen_stats
    return report


def _humanize_eval(df: pd.DataFrame, n: int) -> dict:
    ai = df[(df["label"] == "ai") & df["ai_percent"].notna()].copy()
    if not len(ai) or n <= 0:
        return {"n": 0}
    picks = []
    per_bucket = max(1, n // len(WORD_BUCKETS))
    for name, low, high in WORD_BUCKETS:
        pool = ai[(ai["words"] >= low) & (ai["words"] < high)]
        if len(pool):
            picks.append(pool.sample(n=min(per_bucket, len(pool)), random_state=11))
    sample = pd.concat(picks).drop_duplicates(subset=["id"]).head(n)
    print(f"humanize eval: n={len(sample)} (casual/medium/external)")

    rows = []
    started = time.perf_counter()
    for i, row in enumerate(sample.itertuples(), 1):
        try:
            res = humanize(str(row.text), tone="casual", strength="medium", target="external")
        except Exception as exc:
            rows.append({"id": str(row.id), "error": str(exc)})
            print(f"  [{i}/{len(sample)}] id={row.id} ERROR: {exc}", flush=True)
            continue
        stealth = res.get("stealth") or {}
        before_st = stealth.get("before") or {}
        after_st = stealth.get("after") or {}
        rows.append(
            {
                "id": str(row.id),
                "words": int(row.words),
                "before_ai": res["before"]["ai_percent"],
                "after_ai": res["after"]["ai_percent"],
                "before_zg": res["zerogpt"]["before"],
                "after_zg": res["zerogpt"]["after"],
                "before_risk": before_st.get("risk"),
                "after_risk": after_st.get("risk"),
                "before_worst": before_st.get("worst"),
                "after_worst": after_st.get("worst"),
                "warnings": len(res.get("warnings") or []),
                "elapsed_ms": res.get("elapsed_ms"),
            }
        )
        elapsed = time.perf_counter() - started
        eta = (len(sample) - i) * (elapsed / i) if i else 0
        print(
            f"  [{i}/{len(sample)}] id={row.id} local {res['before']['ai_percent']}->"
            f"{res['after']['ai_percent']}%  external worst "
            f"{before_st.get('worst')}->{after_st.get('worst')}  eta {eta/60:4.1f} min",
            flush=True,
        )

    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": len(rows), "errors": len(rows)}

    frame = pd.DataFrame(ok)

    def _stats(series: pd.Series, threshold: Optional[float] = None) -> dict:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if not len(vals):
            return {}
        out = {
            "mean": round(float(vals.mean()), 2),
            "median": round(float(vals.median()), 2),
            "p25": round(float(vals.quantile(0.25)), 2),
            "p75": round(float(vals.quantile(0.75)), 2),
        }
        if threshold is not None:
            out[f"pct_below_{int(threshold)}"] = round(float((vals < threshold).mean()), 4)
        return out

    by_bucket = {}
    for name, low, high in WORD_BUCKETS:
        sub = frame[(frame["words"] >= low) & (frame["words"] < high)]
        if not len(sub):
            continue
        by_bucket[name] = {
            "n": int(len(sub)),
            "after_local": _stats(sub["after_ai"], 60),
            "after_external_worst": _stats(sub["after_worst"], 50),
            "external_worst_below_20": round(
                float((pd.to_numeric(sub["after_worst"], errors="coerce") < 20).mean()), 4
            ),
        }

    return {
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "local_before": _stats(frame["before_ai"], 60),
        "local_after": _stats(frame["after_ai"], 60),
        "zerogpt_style_before": _stats(frame["before_zg"], 50),
        "zerogpt_style_after": _stats(frame["after_zg"], 50),
        "external_risk_before": _stats(frame["before_risk"], 50),
        "external_risk_after": _stats(frame["after_risk"], 50),
        "external_worst_before": _stats(frame["before_worst"], 50),
        "external_worst_after": _stats(frame["after_worst"], 50),
        "external_worst_below_20": round(
            float((pd.to_numeric(frame["after_worst"], errors="coerce") < 20).mean()), 4
        ),
        "local_after_below_25": round(
            float((pd.to_numeric(frame["after_ai"], errors="coerce") < 25).mean()), 4
        ),
        "elapsed_ms_mean": round(float(pd.to_numeric(frame["elapsed_ms"], errors="coerce").mean()), 1),
        "by_words": by_bucket,
        "samples": ok,
    }


def _fmt_row(label: str, entry: dict) -> str:
    keys = ("n", "mean", "median", "pct_below_60", "pct_below_50", "pct_below_25")
    cells = [f"{label:24s}"]
    for key in keys:
        val = entry.get(key)
        cells.append(f"{str(val) if val is not None else '-':>10s}")
    return " ".join(cells)


def _print_report(report: dict) -> None:
    det = report["detector"]
    print("\n" + "=" * 78)
    print("DETECTOR — binary human vs ai")
    print("=" * 78)
    print(f"samples: {det['n_binary']}  (human {det['n_human']} / ai {det['n_ai']})")
    if "auc" in det:
        print(f"AUC: {det['auc']}   average precision: {det['average_precision']}")
    print("\nthreshold   precision  recall    f1      fpr     fnr    accuracy")
    for row in det.get("thresholds", []):
        marker = "  <- current" if row["threshold"] == det["current_thresholds"]["ai_verdict"] else ""
        print(
            f"{row['threshold']:>6}     {row['precision']:>8} {row['recall']:>7} "
            f"{row['f1']:>7} {row['fpr']:>7} {row['fnr']:>7} {row['accuracy']:>9}{marker}"
        )
    if "optimal" in det:
        opt = det["optimal"]
        print(
            f"\nbest F1 threshold: {opt['f1_max']['threshold']} (F1={opt['f1_max']['f1']})   "
            f"best Youden: {opt['youden_max']['threshold']} (J={opt['youden_max']['j']})"
        )

    print("\nby label:")
    for label, entry in det.get("by_label", {}).items():
        print(
            f"  {label:12s} n={entry['n']:<5} mean={entry['mean_ai_percent']:>6}%  "
            f"flagged_ai@60={entry['flagged_ai_60']:.1%}  not_human@25={entry['not_human_25']:.1%}"
        )

    print("\nby length:")
    for name, entry in det.get("by_words", {}).items():
        parts = [f"  {name:14s} n={entry['n']:<5}"]
        if "auc" in entry:
            parts.append(f"auc={entry['auc']}")
        if "human_mean" in entry:
            parts.append(f"human={entry['human_mean']}%")
        if "ai_mean" in entry:
            parts.append(f"ai={entry['ai_mean']}%")
        if "human_flagged_60" in entry:
            parts.append(f"human_flag={entry['human_flagged_60']:.1%}")
        if "ai_recall_60" in entry:
            parts.append(f"ai_recall={entry['ai_recall_60']:.1%}")
        print(" ".join(parts))

    if det.get("by_generator"):
        print("\nby generator (ai only):")
        for gen, entry in det["by_generator"].items():
            print(
                f"  {gen[:28]:28s} n={entry['n']:<4} mean={entry['mean_ai_percent']:>6}%  "
                f"recall@60={entry['recall_60']:.1%}"
            )

    hum = report.get("humanize") or {}
    if hum.get("n"):
        print("\n" + "=" * 78)
        print(f"HUMANIZE — n={hum['n']} (casual/medium/external, errors={hum.get('errors', 0)})")
        print("=" * 78)
        print(f"local classifier   before mean={hum['local_before'].get('mean')}%  "
              f"after mean={hum['local_after'].get('mean')}%  "
              f"below25={hum['local_after_below_25']:.1%}")
        print(f"external worst     before mean={hum['external_worst_before'].get('mean')}  "
              f"after mean={hum['external_worst_after'].get('mean')}  "
              f"below50={hum['external_worst_after'].get('pct_below_50')}  "
              f"below20={hum['external_worst_below_20']}")
        print(f"external risk      before mean={hum['external_risk_before'].get('mean')}  "
              f"after mean={hum['external_risk_after'].get('mean')}")
        print(f"zerogpt-style pct  before mean={hum['zerogpt_style_before'].get('mean')}  "
              f"after mean={hum['zerogpt_style_after'].get('mean')}")
        print(f"avg humanize time: {hum.get('elapsed_ms_mean')} ms")
        for name, entry in hum.get("by_words", {}).items():
            print(
                f"  {name:14s} n={entry['n']:<3} local_after={entry['after_local'].get('mean')}%  "
                f"external_worst={entry['after_external_worst'].get('mean')}  "
                f"below50={entry['after_external_worst'].get('pct_below_50')}  "
                f"below20={entry['external_worst_below_20']}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=None, help="explicit path to benchmark.parquet")
    parser.add_argument("--limit", type=int, default=None, help="stratified sample size (smoke test)")
    parser.add_argument("--fresh", action="store_true", help="ignore cached score csv")
    parser.add_argument("--skip-humanize", action="store_true", help="detector metrics only")
    parser.add_argument("--humanize-n", type=int, default=50, help="humanize sample size")
    parser.add_argument("--out", default=str(REPORT_JSON), help="report json path")
    args = parser.parse_args()

    started = time.perf_counter()
    df = _load_dataset(args.parquet, args.limit)
    scored = _score_dataset(df, args.fresh)

    report = {
        "parquet_samples": int(len(df)),
        "detector": _detector_report(scored),
        "elapsed_scores": round(time.perf_counter() - started, 1),
    }
    if not args.skip_humanize:
        report["humanize"] = _humanize_eval(scored, args.humanize_n)
    report["elapsed_total"] = round(time.perf_counter() - started, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"report: {out_path}")
    print(f"total time: {report['elapsed_total']}s")


if __name__ == "__main__":
    main()
