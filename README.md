# AI Content Detector

Paste text (or upload PDF/DOCX/TXT) → get an AI-likelihood score, verdict, and sentence-level highlighting. Or switch to the **AI-free** tab to rewrite AI-sounding text so it reads human. Runs 100% locally — FastAPI backend + React frontend, no API keys, no internet needed after setup.

## Quick start (Windows)

Double-click **`run.bat`**. First run installs everything and exports the ONNX model (few minutes); later runs start in ~10 seconds. Browser opens `http://localhost:5173`.

To stop: close the two command windows ("Backend" and "Frontend").

Manual equivalent:

```bat
call .venv\Scripts\activate.bat
start "backend"  cmd /k backend\start_backend.bat     REM port 8000
start "frontend" cmd /k frontend\start_frontend.bat   REM port 5173
```

## Features

- Text paste + file upload (PDF / DOCX / TXT), up to 20k chars analyzed
- AI % (calibrated), verdict (`ai` / `mixed` / `human`), per-sentence scores with highlighting
- **ZeroGPT-style signal**: a second, local estimate (distilgpt2 perplexity + burstiness) shown next to the classifier score — tools like ZeroGPT often disagree with classifier-based detectors, and this gives a rough read on that other signal
- **AI-free rewrite**: choose an external-detector target (default) or this detector, plus casual/formal tone and light/medium/strong strength; get a rewritten version with before/after scores, external-risk estimate, change list, copy/download. External mode preserves headings, lists, tables and paragraph layout
- Scan history (SQLite, local only) — detection scans only, rewrites are not stored
- REST API

## Model

- Base: [`wasitaigeneratedcom/ai-text-detector-small`](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small) (tropa-mini, DeBERTa-v3-large, Apache-2.0) in `models/detector/`
- Scoring: `ai_score = 1 - P(human)` over 4 classes, logit-space calibrated so the published 0.5%-FPR operating point displays as 70%
- Runtime: **ONNX fp32** (`models/detector-onnx/model_fp32.onnx`, exported by `python -m ml.export_onnx`) via `onnxruntime` — ~3× faster than torch on CPU. Torch remains as fallback.
- `DETECTOR_BACKEND=auto|onnx|torch` (default `auto` = ONNX when exported, else torch)
- int8 quantization was tested (dynamic + static/calibrated) and **collapses accuracy** on this DeBERTa-v3 detector, so fp32 ONNX is the serving format.

### Optional disk slimming

After verifying ONNX works (`python -m ml.demo` with `DETECTOR_BACKEND=onnx`), you can delete `models/detector/model.safetensors` (~1.7 GB) — serving uses the ONNX file. Re-export would re-download the weights.

## ZeroGPT-style signal

ZeroGPT-style tools score text by **perplexity + burstiness**, which is a different method from our classifier — so the two disagree, sometimes a lot (marketing copy, headings, and lists often score high on perplexity-based tools while the classifier stays low).

- `ml/zerogpt_style.py` extracts distilgpt2 token-NLL features (mean/std/percentiles, sentence-level variation, burstiness, type-token ratio, heading/list/punctuation ratios) and applies a linear calibrator trained on the public [`wasitaigeneratedcom/ai-detector-benchmark-2026-09`](https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09) dataset (1,595 English samples with ZeroGPT scores).
- Held-out metrics: Spearman **0.55** vs ZeroGPT percent, AUC **0.85** for ZeroGPT percent ≥ 50 (n=1,595). The estimate is *approximate* — it mimics the signal, not the exact tool.
- Coefficients live in `models/zerogpt_style.json`; retrain with `python -m ml._zg_eval` (full feature pass, ~40 min CPU) or refit from a cached CSV (`python -m ml._zg_eval --from-csv models/zg_features.csv --save models/zerogpt_style.json`).
- distilgpt2 (~350 MB) loads lazily on first use; `/api/detect` and `/api/humanize` responses include a `zerogpt` object (`{"pct": 0-100, "prob": 0-1}` for detect, `{"before": ..., "after": ...}` for humanize).

## External-detector mode

`target="external"` is the default for the API and UI. It ranks rewrites against local proxies for the tools users actually check: ZeroGPT, GPTZero and Pangram.

- `ml/stealth.py` combines the distilgpt2 LM features with 42 style/formatting features (sentence rhythm, repetition, contractions, punctuation, paragraph/list structure, readability, hedging, discourse markers) and applies linear proxies from `models/stealth_proxies.json`.
- Proxies are fitted offline on the full-text rows of the public [`wasitaigeneratedcom/ai-detector-benchmark-2026-09`](https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09) dataset (1,595 rows).
- Held-out CV: ZeroGPT Spearman **0.61** / AUC **0.88**; GPTZero AUC **0.80** (n=615); Pangram Spearman **0.71** / AUC **0.85** (n=317); humanized-vs-human AUC **0.91** (advisory only); tropa-3 Spearman **0.69** (advisory only).
- Ensemble weights are tuned out-of-fold: ZeroGPT **0.50**, GPTZero **0.30**, Pangram **0.20**. Candidates are ranked by the worst single proxy first, then weighted risk, with the local classifier as tie-break, so one stubborn tool cannot hide behind an average.
- Responses add `target` and `stealth: {before, after}` with `risk` (weighted 0–100), `worst` (highest of the three tool proxies) and per-tool `components`.
- Layout is preserved: headings, bullet/numbered lists, tables, code fences and blank lines are left intact; only prose lines are rewritten.
- If the distilgpt2 calibrator is missing, external mode falls back to the local detector with a warning.
- Retrain with `python -m ml.stealth_eval` (~1–2 min; writes `models/stealth_features.csv` and `models/stealth_proxies.json`) or refit from the feature cache with `python -m ml.stealth_eval --from-csv models/stealth_features.csv`.
- Reality check: these are *proxy estimates* trained on public labels, not the live tools. Real tool results can differ, and no humanizer can guarantee 0% everywhere.

## AI-free rewrite (humanizer)

Rule-based rewriter in `ml/humanize.py` — no LLM calls, everything local.

- **Target**: `external` (default; ranks candidates by external-detector proxies, conservative paraphrasing, layout-preserving) or `local` (tuned to this detector only).
- **Tone**: `casual` (contractions, simpler words, sentence splitting) or `formal` (keeps a professional register). External mode never adds lowercase style, personal openers or casual endings.
- **Strength**: `light` (word swaps only), `medium`, `strong` (more splitting, list breaks, contract variants). External mode adds candidates with/without contractions and different sentence splits, then picks the lowest-risk rewrite.
- It generates several candidate rewrites and scores each with the detector (batched) plus the ZeroGPT-style signal, then returns the best one. External mode also scores candidates with the stealth proxies first; response includes before/after scores, external risk, a change list, and warnings.
- When the classifier already says human (`< 25%`) but the ZeroGPT-style estimate is high, the **local** target switches to a ZeroGPT-targeted mode: **all** candidates plus the aggressive refinement pass are ranked by the ZeroGPT-style estimate (not just the top-3 by classifier), and a rewrite is returned even if the estimate cannot be lowered (with a warning).
- Typical result on AI-style paragraphs: **~99% → ~0–3%** with casual tone (local classifier). External risk on the same text usually drops to the teens with casual tone; formal tone stays higher because the formal register itself reads AI-like.
- Caveats: `formal` tone keeps the original register, which detectors tend to flag; rewritten text may need a light human pass for flow; on slow CPUs a rewrite can take ~10–20 s (the ZeroGPT-style pass adds distilgpt2 inference).

### Experimental: runtime paraphrase bake-off

A lightweight T5 paraphraser (`humarin/chatgpt_paraphraser_on_T5_base`) was evaluated offline against the rule-based external mode on a 6-text, 90–220-word public-benchmark sample (`models/para_eval.csv`).

| variant | mean external risk | mean worst proxy | mean local classifier |
|---|---|---|---|
| original | 42.1% | 64.5% | 67.2% |
| T5 paraphrase only | 29.2% | 48.4% | 73.1% |
| rule-based external | **27.5%** | 48.5% | **45.0%** |

T5 was not enabled at runtime: it was slower (~25–40 s per 130 words on CPU), sometimes changed names/instructions, and left the local classifier higher. The rule-based layout-preserving engine remains the default; the model stays cached only for experimentation.

## API

| Method | Path | Body |
|---|---|---|
| POST | `/api/detect` | `{"text": "..."}` |
| POST | `/api/detect/file` | multipart file |
| POST | `/api/humanize` | `{"text": "...", "tone": "casual\|formal", "strength": "light\|medium\|strong", "target": "external\|local"}` |
| GET | `/api/history` | — |
| GET/DELETE | `/api/history/{id}` | — |
| GET | `/api/health` | — |

## Tests

```bat
cd backend && ..\.venv\Scripts\python.exe -m pytest -q
```

## Fine-tuning (future, not done)

`ml/prepare_hc3.py` (dataset prep) and `ml/train.py` (CPU fine-tune) are kept for a possible future fine-tune on HC3. Skipped for now — the baseline detector already scores well and CPU training was killed by RAM limits on this machine.

## Layout

```
backend/    FastAPI app (routers, SQLite history, file extraction)
frontend/   Vite + React + Tailwind UI
ml/         model loading, detection pipeline, humanizer, stealth proxies, ONNX export, training scripts
models/     detector weights + ONNX export
data/       local SQLite DB lives in backend/data/
```
