"""Fit local proxies for external detectors from the public benchmark.

Features = distilgpt2 LM features (models/zg_features.csv) + text style
features (ml.stealth). Targets = ZeroGPT percent, GPTZero label, Pangram
score, tropa-3 score, humanized-vs-human label. Artifacts:
models/stealth_features.csv and models/stealth_proxies.json.

Usage:
    python -m ml.stealth_eval
    python -m ml.stealth_eval --from-csv models/stealth_features.csv
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.stealth import STYLE_FEATURE_NAMES, style_features

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LM_CSV = MODELS / "zg_features.csv"
FEATURES_CSV = MODELS / "stealth_features.csv"
PROXIES_JSON = MODELS / "stealth_proxies.json"
DATASET_DIR = (
    MODELS
    / "hf-home"
    / "hub"
    / "datasets--wasitaigeneratedcom--ai-detector-benchmark-2026-09"
)
TARGET_COLUMNS = {"pct", "zg_label", "label", "group", "lang"}
MIN_WEIGHT = 0.15


def _find_dataset() -> Path:
    matches = sorted(DATASET_DIR.glob("snapshots/*/data/*.parquet"))
    if matches:
        return matches[-1]
    return DATASET_DIR / "snapshots" / "missing" / "data" / "benchmark.parquet"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p):
    return np.log(np.clip(p, 0.001, 0.999) / (1.0 - np.clip(p, 0.001, 0.999)))


def _expand(values, mask, rows: int) -> np.ndarray:
    full = np.full(rows, np.nan)
    full[np.asarray(mask, dtype=bool)] = values
    return full


def _auc(y, scores) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _spearman(y, scores) -> float:
    return float(spearmanr(y, scores)[0])


def _build_frame(dataset: Path, lm_csv: Path) -> pd.DataFrame:
    frame = pd.read_parquet(dataset)
    frame = frame.loc[frame["text_available"] == True].reset_index(drop=True)
    lm = pd.read_csv(lm_csv)
    if len(frame) != len(lm):
        raise SystemExit(
            f"row mismatch: dataset {len(frame)} vs {lm_csv.name} {len(lm)}"
        )
    for column in ("label", "group"):
        left = frame[column].astype(str).to_numpy()
        right = lm[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise SystemExit(f"alignment failed on column {column!r}")
    if not np.allclose(
        lm["pct"].to_numpy(dtype=float),
        frame["zerogpt_ai_pct"].to_numpy(dtype=float),
        atol=0.2,
    ):
        raise SystemExit("alignment failed on ZeroGPT percent")

    lm_features = [c for c in lm.columns if c not in TARGET_COLUMNS]
    style = pd.DataFrame(
        [style_features(text) for text in frame["text"].astype(str)], columns=STYLE_FEATURE_NAMES
    )
    out = pd.concat(
        [lm[lm_features].reset_index(drop=True), style.reset_index(drop=True)], axis=1
    )
    out.insert(0, "id", frame["id"].to_numpy())
    out["label"] = frame["label"].to_numpy()
    out["group"] = frame["group"].to_numpy()
    out["lang"] = frame["lang"].to_numpy()
    out["target_zerogpt"] = frame["zerogpt_ai_pct"].to_numpy(dtype=float)
    out["target_gptzero"] = frame["gptzero_label"].to_numpy()
    out["target_pangram"] = frame["pangram_ai_score"].to_numpy(dtype=float)
    out["target_tropa3"] = frame["tropa3_score"].to_numpy(dtype=float)
    feature_cols = lm_features + STYLE_FEATURE_NAMES
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


def _make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=alpha, random_state=42))]
    )


def _make_logistic(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _export(pipe: Pipeline, kind: str, link: str, output: str) -> dict:
    scaler = pipe.named_steps["scale"]
    model = pipe.named_steps["model"]
    return {
        "kind": kind,
        "link": link,
        "output": output,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_.ravel().tolist(),
        "intercept": float(np.ravel(model.intercept_)[0]),
    }


def _ridge_cv(X, y, alpha, link, output):
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for train_idx, test_idx in folds.split(X):
        pipe = _make_ridge(alpha)
        pipe.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict(X[test_idx])
    final = _make_ridge(alpha)
    final.fit(X, y)
    return _export(final, "ridge", link, output), oof


def _logistic_cv(X, y, c_value):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = np.asarray(y)
    scores = np.zeros(len(y))
    for train_idx, test_idx in folds.split(X, y):
        pipe = _make_logistic(c_value)
        pipe.fit(X[train_idx], y[train_idx])
        scores[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    final = _make_logistic(c_value)
    final.fit(X, y)
    return _export(final, "logistic", "sigmoid", "prob"), scores


def _top_features(names, proxy, limit=8) -> list:
    pairs = sorted(
        zip(names, proxy["coef"]), key=lambda item: abs(item[1]), reverse=True
    )
    return [(name, round(coef, 3)) for name, coef in pairs[:limit]]


def _print_top(names, proxy) -> None:
    top = ", ".join(f"{name}={coef}" for name, coef in _top_features(names, proxy))
    print(f"    top: {top}")


def fit(frame: pd.DataFrame) -> dict:
    feature_names = [
        c for c in frame.columns if not c.startswith("target_") and c not in ("id",)
        and c not in ("label", "group", "lang")
    ]
    X = frame[feature_names].to_numpy(dtype=float)
    metrics = {}
    proxies = {}
    oof = {}

    zg = frame["target_zerogpt"].to_numpy(dtype=float)
    proxy, pred = _ridge_cv(X, zg, alpha=2.0, link="identity", output="pct")
    proxies["zerogpt"] = proxy
    oof["zerogpt"] = np.clip(pred, 0.0, 100.0)
    metrics["zerogpt"] = {
        "spearman": round(_spearman(zg, pred), 3),
        "auc_50": round(_auc((zg >= 50).astype(int), pred), 3),
        "mae": round(float(mean_absolute_error(zg, pred)), 2),
        "n": int(len(zg)),
    }
    print(f"zerogpt  {metrics['zerogpt']}")
    _print_top(feature_names, proxy)

    gz_labels = frame["target_gptzero"].astype(str).str.lower()
    gz_mask = frame["target_gptzero"].notna().to_numpy() & ~gz_labels.isin(
        ["nan", "none", ""]
    ).to_numpy()
    gz_frame = frame.loc[gz_mask]
    gz_y = gz_frame["target_gptzero"].astype(str).str.lower().isin(["ai", "mixed"]).to_numpy(dtype=int)
    proxy, proba = _logistic_cv(X[gz_mask], gz_y, c_value=0.3)
    proxies["gptzero"] = proxy
    oof["gptzero"] = _expand(proba * 100.0, gz_mask, len(frame))
    pure_ai = gz_frame["target_gptzero"].astype(str).str.lower().eq("ai").to_numpy()
    human = gz_frame["target_gptzero"].astype(str).str.lower().eq("human").to_numpy()
    metrics["gptzero"] = {
        "auc_ai_mixed": round(_auc(gz_y, proba), 3),
        "auc_ai_only": round(
            _auc(pure_ai[human | pure_ai], proba[human | pure_ai]), 3
        ),
        "n": int(gz_mask.sum()),
        "n_pos": int(gz_y.sum()),
    }
    print(f"gptzero  {metrics['gptzero']}")
    _print_top(feature_names, proxy)

    pg = frame["target_pangram"].to_numpy(dtype=float)
    pg_mask = frame["target_pangram"].notna().to_numpy() & np.isfinite(pg)
    logit_target = _logit(pg[pg_mask])
    proxy_logit, pred_logit = _ridge_cv(
        X[pg_mask], logit_target, alpha=2.0, link="sigmoid", output="prob"
    )
    proxy_raw, pred_raw = _ridge_cv(
        X[pg_mask], pg[pg_mask], alpha=2.0, link="identity", output="prob"
    )
    score_logit = _spearman(pg[pg_mask], _sigmoid(pred_logit))
    score_raw = _spearman(pg[pg_mask], np.clip(pred_raw, 0.0, 1.0))
    use_logit = score_logit >= score_raw
    proxy, pred = (proxy_logit, _sigmoid(pred_logit)) if use_logit else (
        proxy_raw,
        np.clip(pred_raw, 0.0, 1.0),
    )
    proxies["pangram"] = proxy
    oof["pangram"] = _expand(pred * 100.0, pg_mask, len(frame))
    metrics["pangram"] = {
        "spearman": round(max(score_logit, score_raw), 3),
        "auc_50": round(_auc((pg[pg_mask] >= 0.5).astype(int), pred), 3),
        "target": "logit" if use_logit else "raw",
        "n": int(pg_mask.sum()),
    }
    print(f"pangram  {metrics['pangram']}")
    _print_top(feature_names, proxy)

    hz_mask = frame["label"].isin(["human", "humanized"]).to_numpy()
    hz_y = frame.loc[hz_mask, "label"].eq("humanized").to_numpy(dtype=int)
    proxy, proba = _logistic_cv(X[hz_mask], hz_y, c_value=0.5)
    proxies["humanized"] = proxy
    oof["humanized"] = _expand(proba * 100.0, hz_mask, len(frame))
    metrics["humanized"] = {
        "auc": round(_auc(hz_y, proba), 3),
        "n": int(hz_mask.sum()),
        "n_pos": int(hz_y.sum()),
    }
    print(f"humanized {metrics['humanized']}")
    _print_top(feature_names, proxy)

    tr = frame["target_tropa3"].to_numpy(dtype=float)
    tr_mask = frame["target_tropa3"].notna().to_numpy() & np.isfinite(tr)
    proxy, pred = _ridge_cv(
        X[tr_mask], _logit(tr[tr_mask]), alpha=2.0, link="sigmoid", output="prob"
    )
    proxies["tropa3"] = proxy
    oof["tropa3"] = _expand(_sigmoid(pred) * 100.0, tr_mask, len(frame))
    metrics["tropa3"] = {
        "spearman": round(_spearman(tr[tr_mask], _sigmoid(pred)), 3),
        "auc_50": round(_auc((tr[tr_mask] >= 0.5).astype(int), _sigmoid(pred)), 3),
        "n": int(tr_mask.sum()),
    }
    print(f"tropa3   {metrics['tropa3']}")
    _print_top(feature_names, proxy)

    weights, tuned = _tune_weights(frame, oof)
    ensemble = _ensemble_frame(oof, len(frame), weights)
    metrics["ensemble"] = _ensemble_metrics(frame, ensemble)
    metrics["ensemble"]["tuned"] = {**tuned, "weights": weights}
    print(f"ensemble {metrics['ensemble']['summary']} weights={weights}")

    return {
        "features": feature_names,
        "proxies": proxies,
        "weights": weights,
        "metrics": metrics,
        "rows": int(len(frame)),
        "version": 1,
    }


def _ensemble_frame(oof: dict, rows: int, weights: dict) -> np.ndarray:
    risk = np.zeros(rows)
    weight_total = np.zeros(rows)
    for key, values in oof.items():
        weight = weights.get(key, 0.0)
        mask = np.isfinite(values)
        risk[mask] += values[mask] * weight
        weight_total[mask] += weight
    return np.divide(risk, np.maximum(weight_total, 1e-9))


def _tune_weights(frame: pd.DataFrame, oof: dict) -> tuple:
    keys = ["gptzero", "pangram", "zerogpt"]
    gz = frame["target_gptzero"].astype(str).str.lower()
    known = frame["target_gptzero"].notna().to_numpy() & ~gz.isin(
        ["nan", "none", ""]
    ).to_numpy()
    gz_y = gz[known].isin(["ai", "mixed"]).to_numpy(dtype=int)
    zg = frame["target_zerogpt"].to_numpy(dtype=float)

    best = None
    grid = np.round(np.arange(MIN_WEIGHT, 1.0001, 0.05), 2)
    for combo in product(grid, repeat=len(keys)):
        if abs(float(sum(combo)) - 1.0) > 1e-9:
            continue
        weights = {key: float(weight) for key, weight in zip(keys, combo)}
        weights["humanized"] = 0.0
        weights["tropa3"] = 0.0
        risk = _ensemble_frame(oof, len(frame), weights)
        auc = _auc(gz_y, risk[known])
        rho = _spearman(zg, risk)
        score = 0.5 * auc + 0.5 * rho
        if best is None or score > best[0]:
            best = (score, weights, auc, rho)
    return best[1], {"auc_gptzero": round(best[2], 3), "spearman_zerogpt": round(best[3], 3)}


def _ensemble_metrics(frame: pd.DataFrame, risk: np.ndarray) -> dict:
    gz = frame["target_gptzero"].astype(str).str.lower()
    known = frame["target_gptzero"].notna().to_numpy() & ~gz.isin(
        ["nan", "none", ""]
    ).to_numpy()
    auc = None
    if known.any():
        y = gz[known].isin(["ai", "mixed"]).to_numpy(dtype=int)
        auc = round(_auc(y, risk[known]), 3)
    spearman = round(
        _spearman(frame["target_zerogpt"].to_numpy(dtype=float), risk), 3
    )
    groups = {}
    for name in ("human", "ai", "humanized"):
        mask = frame["label"].eq(name).to_numpy()
        if mask.any():
            groups[name] = round(float(np.mean(risk[mask])), 1)
    summary = {"auc_gptzero": auc, "spearman_zerogpt": spearman, **groups}
    return {"summary": summary}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--lm-csv", type=Path, default=LM_CSV)
    parser.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    parser.add_argument("--save", type=Path, default=PROXIES_JSON)
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
        print(f"loaded features {args.from_csv} ({len(frame)} rows)")
    else:
        dataset = args.dataset or _find_dataset()
        if not dataset.exists():
            print(f"dataset not found: {dataset}", file=sys.stderr)
            return 2
        frame = _build_frame(dataset, args.lm_csv)
        frame.to_csv(args.features_csv, index=False)
        print(f"wrote features {args.features_csv} ({len(frame)} rows)")

    data = fit(frame)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    with args.save.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote proxies {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
