"""Export the detector to ONNX (fp32) and verify parity with torch.

Run once (needs torch + onnxscript + onnxruntime):
    python -m ml.export_onnx

Output:
    models/detector-onnx/model_fp32.onnx   (serving model)

Note: int8 quantization was tested (dynamic per-tensor/per-channel, static
calibrated QDQ) and collapses accuracy on this DeBERTa-v3 detector (ai_score
drift up to 0.8), so fp32 ONNX is the serving format. ORT still beats torch
eager on CPU latency and loads without the torch runtime.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from ml import config as cfg
from ml._torch_model import load_torch_model

SAMPLES = [
    # human
    "Hey, did you watch the match last night? That last-over finish was crazy.",
    "Please find attached the quarterly report. Let me know if you need the raw data.",
    "The train arrives at 6:45 platform 3, so reach the station by 6:30 at least.",
    # ai
    "Artificial intelligence has revolutionized numerous industries by enabling "
    "unprecedented efficiency and data-driven decision-making processes.",
    "In conclusion, it is important to note that the implementation of "
    "comprehensive strategies yields optimal outcomes across multiple domains.",
    "Certainly! Here is a detailed overview of the key considerations: first, "
    "performance metrics; second, scalability; and third, long-term maintainability.",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def export() -> Path:
    source = cfg.resolve_model_source()
    print(f"Loading torch model from {source} ...")
    config = AutoConfig.from_pretrained(source)
    model, _ = load_torch_model(source, config)
    model.eval()

    cfg.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = cfg.ONNX_DIR / "model_fp32.onnx"

    dummy_ids = torch.ones(2, 64, dtype=torch.long)
    dummy_mask = torch.ones(2, 64, dtype=torch.long)

    print("Exporting fp32 ONNX ...")
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"  -> {fp32_path} ({fp32_path.stat().st_size / 1e6:.0f} MB)")
    return fp32_path


def _probs_onnx(path: Path, texts: list, tokenizer, max_length: int) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, min(4, torch.get_num_threads()))
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    enc = tokenizer(
        texts, truncation=True, max_length=max_length, add_special_tokens=True, padding=True
    )
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    mask = np.asarray(enc["attention_mask"], dtype=np.int64)
    t0 = time.perf_counter()
    logits = sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
    dt = time.perf_counter() - t0
    print(f"  onnx batch of {len(texts)}: {dt*1000:.0f} ms")
    return _softmax(logits.astype(np.float32))


def _probs_torch(texts: list, tokenizer, max_length: int) -> np.ndarray:
    source = cfg.resolve_model_source()
    config = AutoConfig.from_pretrained(source)
    model, _ = load_torch_model(source, config)
    model.eval()
    enc = tokenizer(
        texts, truncation=True, max_length=max_length, add_special_tokens=True, padding=True
    )
    ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    with torch.inference_mode():
        t0 = time.perf_counter()
        logits = model(input_ids=ids, attention_mask=mask)
        dt = time.perf_counter() - t0
    print(f"  torch batch of {len(texts)}: {dt*1000:.0f} ms")
    return _softmax(logits.float().numpy())


def parity(fp32_path: Path) -> bool:
    from ml.detect import calibrate

    source = cfg.resolve_model_source()
    tokenizer = AutoTokenizer.from_pretrained(source)
    print("\nParity check (torch vs onnx fp32):")
    p_torch = _probs_torch(SAMPLES, tokenizer, cfg.SENTENCE_MAX_LENGTH)
    p_onnx = _probs_onnx(fp32_path, SAMPLES, tokenizer, cfg.SENTENCE_MAX_LENGTH)

    print(f"  {'sample':<10} {'torch':>8} {'onnx':>8}   (ai_score)")
    max_diff = 0.0
    ok = True
    for i, _ in enumerate(SAMPLES):
        ai_t = 1.0 - p_torch[i, 0]
        ai_o = 1.0 - p_onnx[i, 0]
        max_diff = max(max_diff, abs(ai_t - ai_o))
        kind = "human" if i < 3 else "ai"
        print(
            f"  {kind + str(i % 3):<10} {ai_t:8.4f} {ai_o:8.4f}"
            f"   calib: {calibrate(float(ai_t)):5.1f}% -> {calibrate(float(ai_o)):5.1f}%"
        )
        # Gate on agreement with torch (not with the text's known origin).
        torch_human = calibrate(float(ai_t)) < cfg.MIXED_VERDICT_THRESHOLD
        onnx_human = calibrate(float(ai_o)) < cfg.MIXED_VERDICT_THRESHOLD
        if torch_human != onnx_human:
            ok = False
    probs_diff = float(np.abs(p_torch - p_onnx).max())
    print(f"  max |torch-onnx| probs diff: {probs_diff:.6f}")
    if max_diff > 0.01:
        print("  FAIL: onnx diverges from torch")
        ok = False
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    if "--parity-only" in sys.argv:
        fp32_path = cfg.ONNX_DIR / "model_fp32.onnx"
        if not fp32_path.exists():
            print("model_fp32.onnx missing — run without --parity-only first")
            return 1
    else:
        fp32_path = export()
    if not parity(fp32_path):
        return 1
    print(f"\nDone. Serving model: {fp32_path}")
    print("Smoke test: set DETECTOR_BACKEND=onnx then python -m ml.demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
