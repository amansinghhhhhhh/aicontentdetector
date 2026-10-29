import Gauge from "./Gauge.jsx";
import { sentenceClass, verdictMeta } from "../lib/styles.js";

const CLASS_ORDER = ["human", "ai", "ai_edited", "humanized"];
const CLASS_COLORS = {
  human: "bg-emerald-500",
  ai: "bg-rose-500",
  ai_edited: "bg-orange-400",
  humanized: "bg-fuchsia-500",
};

function Stat({ label, value }) {
  return (
    <div className="flex-1 rounded-xl bg-slate-950/60 px-3 py-2.5 text-center ring-1 ring-slate-800">
      <div className="text-lg font-bold tabular-nums">{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  );
}

export default function ResultPanel({ result }) {
  if (!result) return null;
  const meta = verdictMeta(result.verdict);
  const stats = result.stats;
  const classEntries = CLASS_ORDER.map((key) => ({
    key,
    value: result.class_percent?.[key] ?? 0,
  }));
  const hasMix = classEntries.some((c) => c.value > 0 && c.value < 99.5);

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-xl shadow-black/20">
      <div className="grid gap-6 md:grid-cols-[240px_1fr] md:items-start">
        <div className="flex justify-center">
          <Gauge
            percent={result.ai_percent}
            verdict={result.verdict}
            confidence={result.confidence}
          />
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="Words" value={stats.words} />
            <Stat label="Sentences" value={stats.sentences} />
            <Stat label="AI sentences" value={stats.ai_sentences} />
            <Stat label="Scanned in" value={`${(result.elapsed_ms / 1000).toFixed(1)}s`} />
          </div>

          <div className="rounded-xl bg-slate-950/60 p-3.5 ring-1 ring-slate-800">
            <div className="mb-2 flex items-center justify-between text-[11px] text-slate-400">
              <span className="font-semibold uppercase tracking-wider">Composition</span>
              <span className="text-slate-600">model class probabilities</span>
            </div>
            <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
              {classEntries.map(({ key, value }) =>
                value > 0.4 ? (
                  <div
                    key={key}
                    className={`${CLASS_COLORS[key]} transition-all duration-700`}
                    style={{ width: `${value}%` }}
                    title={`${key}: ${value.toFixed(1)}%`}
                  />
                ) : null
              )}
            </div>
            {hasMix && (
              <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-slate-400">
                {classEntries.map(({ key, value }) =>
                  value > 0.4 ? (
                    <span key={key} className="inline-flex items-center gap-1.5">
                      <i className={`h-2 w-2 rounded-full ${CLASS_COLORS[key]}`} />
                      {key.replace("_", "-")} {value.toFixed(1)}%
                    </span>
                  ) : null
                )}
              </div>
            )}
          </div>

          {result.zerogpt && (
            <div className="rounded-xl bg-slate-950/60 p-3.5 ring-1 ring-slate-800">
              <div className="mb-1.5 flex items-center justify-between text-[11px] text-slate-400">
                <span className="font-semibold uppercase tracking-wider">
                  ZeroGPT-style signal
                </span>
                <span className="text-slate-600">perplexity + burstiness</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-2xl font-extrabold tabular-nums ${
                    result.zerogpt.pct >= 60
                      ? "text-rose-400"
                      : result.zerogpt.pct >= 25
                        ? "text-amber-300"
                        : "text-emerald-400"
                  }`}
                >
                  {result.zerogpt.pct.toFixed(1)}
                  <span className="text-sm font-bold text-slate-500">%</span>
                </span>
                <span className="text-xs text-slate-500">AI-likelihood estimate</span>
              </div>
              <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
                Local approximation of the signal ZeroGPT-style tools use. It can
                disagree with the classifier above — headings, lists, and marketing
                copy often score high here.
              </p>
            </div>
          )}

          {result.warnings?.length > 0 && (
            <div className="space-y-2">
              {result.warnings.map((w) => (
                <p
                  key={w}
                  className="rounded-lg bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-300 ring-1 ring-inset ring-amber-500/20"
                >
                  {w}
                </p>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="mt-6">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-200">
            Sentence-by-sentence analysis
          </h3>
          <div className="flex gap-3 text-[11px] text-slate-500">
            <span className="inline-flex items-center gap-1.5">
              <i className="h-2 w-2 rounded-full bg-emerald-500" /> human
            </span>
            <span className="inline-flex items-center gap-1.5">
              <i className="h-2 w-2 rounded-full bg-amber-500" /> uncertain
            </span>
            <span className="inline-flex items-center gap-1.5">
              <i className="h-2 w-2 rounded-full bg-rose-500" /> AI
            </span>
          </div>
        </div>

        <div className="rounded-xl bg-slate-950/60 p-4 leading-loose ring-1 ring-slate-800">
          {result.sentences.map((s) => (
            <span
              key={s.index}
              className={`group relative mx-[1px] my-[1px] cursor-default rounded px-1 py-0.5 transition ${sentenceClass(
                s.score
              )}`}
              title={`AI score: ${s.score}%`}
            >
              {s.text}
              <span
                className={`ml-1.5 hidden align-middle text-[10px] font-bold tabular-nums sm:inline ${
                  s.score >= 60
                    ? "text-rose-300"
                    : s.score >= 25
                      ? "text-amber-300"
                      : "text-emerald-300"
                }`}
              >
                <span className="opacity-0 transition group-hover:opacity-100">
                  {s.score}%
                </span>
              </span>
            </span>
          ))}
        </div>

        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-slate-600">
          {result.source && <span>source: {result.source}</span>}
          {result.filename && <span>file: {result.filename}</span>}
          {result.id && <span>scan #{result.id}</span>}
          <span>chunks: {stats.chunks}</span>
        </div>
      </div>
    </section>
  );
}
