"""Experiment: can local perplexity/structure features predict ZeroGPT scores?

Downloads the public ZeroGPT-labeled benchmark, extracts features via
ml.zerogpt_style, and reports cross-validated correlation / AUC; can fit and
persist the linear calibrator to models/zerogpt_style.json. Not used at
runtime.

python -m ml._zg_eval [--rows N] [--out CSV] [--save JSON] [--extra FILE ...]
python -m ml._zg_eval --from-csv models/zg_features.csv --save models/zerogpt_style.json
"""

import argparse
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

from ml.zerogpt_style import MAX_TOKENS, MODEL_ID, extract_features

LABEL_COLS = {"pct", "zg_label", "label", "group", "lang"}


def run_extraction(args):
    import pandas as pd

    print("loading dataset...", flush=True)
    ds = load_dataset(
        "wasitaigeneratedcom/ai-detector-benchmark-2026-09", split="train"
    )
    rows = [
        {
            "text": r["text"],
            "pct": r["zerogpt_ai_pct"],
            "zg_label": r["zerogpt_label"],
            "label": r["label"],
            "group": r["group"],
            "lang": r["lang"],
        }
        for r in ds
        if r["text_available"] and r["text"] and r["zerogpt_ai_pct"] is not None
    ]
    print(f"usable rows: {len(rows)}", flush=True)
    if args.rows and args.rows < len(rows):
        rng = np.random.default_rng(7)
        picked = rng.choice(len(rows), size=args.rows, replace=False)
        rows = [rows[i] for i in sorted(picked)]
        print(f"sampled {len(rows)} rows (seed 7)", flush=True)

    started = time.perf_counter()
    records = []
    for i, row in enumerate(rows):
        try:
            feats = extract_features(row["text"])
        except Exception as exc:
            print(f"  skip row {i}: {exc}", flush=True)
            continue
        if feats is None:
            print(f"  skip row {i}: empty", flush=True)
            continue
        feats.update(
            {
                "pct": row["pct"],
                "zg_label": row["zg_label"],
                "label": row["label"],
                "group": row["group"],
                "lang": row["lang"],
            }
        )
        records.append(feats)
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - started
            print(f"  {i + 1}/{len(rows)} ({elapsed:.0f}s)", flush=True)

    frame = pd.DataFrame(records)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows) in {time.perf_counter() - started:.0f}s")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=0, help="limit rows for a quick run")
    parser.add_argument("--out", default="models/zg_features.csv")
    parser.add_argument("--extra", nargs="*", default=[], help="extra text files to score")
    parser.add_argument("--save", default="", help="write fitted linear calibrator JSON here")
    parser.add_argument("--from-csv", default="", help="skip extraction, reuse a features CSV")
    args = parser.parse_args()

    if args.from_csv:
        import pandas as pd

        frame = pd.read_csv(args.from_csv)
        print(f"loaded {args.from_csv} ({len(frame)} rows)", flush=True)
    else:
        frame = run_extraction(args)

    feature_cols = [c for c in frame.columns if c not in LABEL_COLS]
    from scipy.stats import spearmanr
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = frame[feature_cols].fillna(0.0).to_numpy()
    y_pct = frame["pct"].to_numpy()
    y_ai = (frame["zg_label"] == "ai").astype(int).to_numpy()
    y_aiish = (frame["pct"] >= 50).astype(int).to_numpy()

    cv = KFold(5, shuffle=True, random_state=0)
    ridge = make_pipeline(StandardScaler(), Ridge())
    rho = spearmanr(
        cross_val_predict(ridge, x, y_pct, cv=cv).ravel(), y_pct
    ).statistic
    print(f"ridge vs zerogpt_pct: spearman={rho:.3f}")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    proba = cross_val_predict(clf, x, y_aiish, cv=cv, method="predict_proba")[:, 1]
    auc50 = roc_auc_score(y_aiish, proba)
    print(f"logreg vs zerogpt_pct>=50: auc={auc50:.3f} (pos={y_aiish.sum()}/{len(y_aiish)})")

    if y_ai.sum() >= 10:
        proba_ai = cross_val_predict(clf, x, y_ai, cv=cv, method="predict_proba")[:, 1]
        print(
            f"logreg vs zerogpt_label==ai: auc={roc_auc_score(y_ai, proba_ai):.3f} "
            f"(pos={y_ai.sum()}/{len(y_ai)})"
        )

    clf.fit(x, y_aiish)
    coefs = sorted(
        zip(feature_cols, clf[-1].coef_[0]),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    print("top logreg coefficients (standardized):")
    for name, weight in coefs[:12]:
        print(f"  {name:>22}: {weight:+.3f}")

    for name, target in (("pct", y_pct), ("ai50", y_aiish)):
        model_hgb = (
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)
            if name == "pct"
            else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)
        )
        pred = cross_val_predict(
            model_hgb,
            x,
            target,
            cv=cv,
            method="predict" if name == "pct" else "predict_proba",
        )
        if name == "pct":
            print(f"hgb vs zerogpt_pct: spearman={spearmanr(pred, target).statistic:.3f}")
        else:
            print(f"hgb vs zerogpt_pct>=50: auc={roc_auc_score(target, pred[:, 1]):.3f}")

    if args.save or args.extra:
        ridge.fit(x, y_pct)

    if args.save:
        import json

        payload = {
            "features": feature_cols,
            "logreg": {
                "mean": clf[0].mean_.tolist(),
                "scale": clf[0].scale_.tolist(),
                "coef": clf[-1].coef_[0].tolist(),
                "intercept": float(clf[-1].intercept_[0]),
                "target": "zerogpt_pct>=50",
                "cv_auc": float(auc50),
            },
            "ridge_pct": {
                "mean": ridge[0].mean_.tolist(),
                "scale": ridge[0].scale_.tolist(),
                "coef": ridge[-1].coef_.tolist(),
                "intercept": float(ridge[-1].intercept_),
                "cv_spearman": float(rho),
            },
            "model_id": MODEL_ID,
            "max_tokens": MAX_TOKENS,
            "rows": int(len(frame)),
        }
        save_path = Path(args.save)
        save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved calibrator -> {save_path}")

    if args.extra:
        print("extra text predictions:")
        for path_str in args.extra:
            path = Path(path_str)
            text = path.read_text(encoding="utf-8")
            feats = extract_features(text)
            if feats is None:
                print(f"  {path.name}: empty")
                continue
            vec = np.array([[feats[c] for c in feature_cols]])
            pct_pred = float(np.clip(ridge.predict(vec)[0], 0, 100))
            ai_pred = float(clf.predict_proba(vec)[0][1] * 100)
            print(f"  {path.name}: ridge_pct={pct_pred:.1f} logreg_ai={ai_pred:.1f}")


if __name__ == "__main__":
    main()
