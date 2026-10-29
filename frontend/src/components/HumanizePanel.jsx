import { useState } from "react";
import { verdictMeta } from "../lib/styles.js";

const STEALTH_LABELS = {
  zerogpt: "ZeroGPT",
  gptzero: "GPTZero",
  pangram: "Pangram",
};

function riskColor(risk) {
  if (risk < 25) return "text-emerald-300";
  if (risk < 50) return "text-amber-300";
  return "text-rose-300";
}

function ScoreCard({ label, score }) {
  const meta = verdictMeta(score.verdict);
  return (
    <div className="flex-1 rounded-xl bg-slate-950/60 px-3 py-3 text-center ring-1 ring-slate-800">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`mt-1 text-3xl font-extrabold tabular-nums ${meta.text}`}>
        {score.ai_percent.toFixed(1)}
        <span className="text-base font-bold text-slate-500">%</span>
      </div>
      <div className={`mt-1 inline-block rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ${meta.chip}`}>
        {meta.label}
      </div>
    </div>
  );
}

export default function HumanizePanel({ result, onUse, onCheck, checking }) {
  const [copied, setCopied] = useState(false);
  if (!result) return null;

  const drop = result.before.ai_percent - result.after.ai_percent;
  const zg = result.zerogpt;
  const stealth = result.target === "external" ? result.stealth : null;
  const stealthBefore = stealth?.before;
  const stealthAfter = stealth?.after;
  const zgFocused = !stealth && result.before.ai_percent < 25 && zg?.before != null;
  const zgBlock =
    zg?.before != null ? (
      <div
        className={
          zgFocused
            ? "mb-4 rounded-xl bg-amber-500/10 px-3 py-2 text-center text-[11px] text-amber-200 ring-1 ring-inset ring-amber-500/20"
            : "mt-2 text-center text-[11px] text-slate-500"
        }
      >
        ZeroGPT-style estimate{zgFocused ? " (target of this rewrite)" : ""}:{" "}
        <span
          className={
            zgFocused
              ? "tabular-nums font-semibold text-amber-100"
              : "tabular-nums text-slate-400"
          }
        >
          {zg.before.toFixed(1)}%
        </span>
        <span className="mx-1.5 text-slate-600">→</span>
        <span
          className={
            zgFocused
              ? "tabular-nums font-semibold text-amber-100"
              : "tabular-nums text-slate-400"
          }
        >
          {zg.after != null ? `${zg.after.toFixed(1)}%` : "—"}
        </span>
      </div>
    ) : null;
  const stealthBlock = stealthBefore ? (
    <div className="mb-4 rounded-xl bg-slate-950/60 px-3 py-3 ring-1 ring-slate-800">
      <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 text-[11px]">
        <span className="font-semibold uppercase tracking-wider text-slate-500">
          Estimated external risk
        </span>
        <span className={`tabular-nums font-bold ${riskColor(stealthBefore.risk)}`}>
          {stealthBefore.risk.toFixed(1)}%
        </span>
        <span className="text-slate-600">→</span>
        <span
          className={`tabular-nums font-bold ${
            stealthAfter ? riskColor(stealthAfter.risk) : "text-slate-500"
          }`}
        >
          {stealthAfter ? `${stealthAfter.risk.toFixed(1)}%` : "—"}
        </span>
      </div>
      {stealthAfter?.worst != null && stealthAfter.worst > stealthAfter.risk && (
        <div className="mt-1 text-center text-[10px] text-slate-500">
          Worst single proxy:{" "}
          <span
            className={`tabular-nums font-semibold ${riskColor(stealthAfter.worst)}`}
          >
            {stealthAfter.worst.toFixed(1)}%
          </span>
        </div>
      )}
      {stealthAfter?.components && (
        <div className="mt-2 flex flex-wrap justify-center gap-1.5">
          {Object.entries(STEALTH_LABELS).map(([key, label]) => {
            const value = stealthAfter.components[key];
            if (value == null) return null;
            return (
              <span
                key={key}
                className="rounded-md bg-slate-900 px-2 py-1 text-[10px] text-slate-400 ring-1 ring-slate-800"
              >
                {label}{" "}
                <span className="tabular-nums font-semibold text-slate-200">
                  {value.toFixed(1)}%
                </span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  ) : null;

  async function copy() {
    try {
      await navigator.clipboard.writeText(result.rewritten);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([result.rewritten], { type: "text/plain" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "ai-free-rewrite.txt";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="rounded-2xl border border-emerald-500/25 bg-slate-900/60 p-5 shadow-xl shadow-black/20">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-200">AI-free rewrite</h3>
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
          <span className="rounded-full bg-slate-800 px-2 py-0.5 ring-1 ring-slate-700">
            {result.tone}
          </span>
          <span className="rounded-full bg-slate-800 px-2 py-0.5 ring-1 ring-slate-700">
            {result.strength}
          </span>
          <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-emerald-300 ring-1 ring-emerald-500/25">
            {stealth ? "external" : "this detector"}
          </span>
          <span className="tabular-nums text-slate-500">
            {(result.elapsed_ms / 1000).toFixed(1)}s
          </span>
        </div>
      </div>

      {stealthBlock}
      {zgFocused && zgBlock}

      <div className="flex items-stretch gap-2 sm:gap-3">
        <ScoreCard label="Before" score={result.before} />
        <div className="flex flex-col items-center justify-center px-1">
          <span className="text-lg text-slate-500">→</span>
          <span
            className={`text-[11px] font-bold tabular-nums ${
              drop > 0 ? "text-emerald-400" : "text-slate-500"
            }`}
          >
            {drop >= 0 ? "−" : "+"}
            {Math.abs(drop).toFixed(1)}
          </span>
        </div>
        <ScoreCard label="After" score={result.after} />
      </div>

      {!stealth && !zgFocused && zgBlock}

      <div className="mt-4 rounded-xl bg-slate-950/60 p-4 ring-1 ring-slate-800">
        <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-slate-100">
          {result.rewritten}
        </p>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <button
          onClick={copy}
          className="rounded-lg px-3 py-2 text-xs font-medium text-slate-300 ring-1 ring-slate-700 transition hover:bg-slate-800"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
        <button
          onClick={download}
          className="rounded-lg px-3 py-2 text-xs font-medium text-slate-300 ring-1 ring-slate-700 transition hover:bg-slate-800"
        >
          Download .txt
        </button>
        <button
          onClick={() => onUse?.(result.rewritten)}
          className="rounded-lg px-3 py-2 text-xs font-medium text-slate-300 ring-1 ring-slate-700 transition hover:bg-slate-800"
        >
          Use as input
        </button>
        <button
          onClick={() => onCheck?.(result.rewritten)}
          disabled={checking}
          className="rounded-lg bg-sky-500/15 px-3 py-2 text-xs font-semibold text-sky-300 ring-1 ring-sky-500/30 transition hover:bg-sky-500/25 disabled:opacity-50"
        >
          {checking ? "Checking…" : "Check with detector"}
        </button>
      </div>

      {result.changes?.length > 0 && (
        <div className="mt-4">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Changes
          </div>
          <div className="flex flex-wrap gap-1.5">
            {result.changes.slice(0, 16).map((change) => (
              <span
                key={`${change.source}-${change.target}`}
                className="inline-flex items-center gap-1 rounded-md bg-slate-950/70 px-2 py-1 text-[11px] text-slate-400 ring-1 ring-slate-800"
              >
                <span className="text-slate-500 line-through">{change.source}</span>
                <span className="text-slate-600">→</span>
                <span className="text-slate-200">{change.target}</span>
                {change.count > 1 && (
                  <span className="tabular-nums text-slate-500">×{change.count}</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {result.warnings?.length > 0 && (
        <div className="mt-4 space-y-2">
          {result.warnings.map((warning) => (
            <p
              key={warning}
              className="rounded-lg bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-300 ring-1 ring-inset ring-amber-500/20"
            >
              {warning}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
