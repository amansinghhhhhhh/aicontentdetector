import { useRef, useState } from "react";
import { countWords } from "../lib/styles.js";

const ANALYSIS_CHAR_LIMIT = 20000;
const MAX_CHARS = 500000;

const TONES = [
  { id: "casual", label: "Casual" },
  { id: "formal", label: "Formal" },
];

const STRENGTHS = [
  { id: "light", label: "Light" },
  { id: "medium", label: "Medium" },
  { id: "strong", label: "Strong" },
];

const TARGETS = [
  { id: "external", label: "External tools" },
  { id: "local", label: "This detector" },
];

const AI_EXAMPLE =
  "In today's fast-paced digital landscape, artificial intelligence has emerged as a transformative force across numerous industries. Organizations are increasingly leveraging AI-driven solutions to streamline operations, enhance decision-making processes, and unlock valuable insights from vast datasets. Moreover, by embracing these cutting-edge technologies, businesses can maintain a competitive edge while delivering superior value to their customers.";

const HUMAN_EXAMPLE =
  "okay so i finally tried that new ramen place downtown and honestly? not worth the hype. the broth was kinda oily and my noodles were overcooked lol. i asked for extra chashu and they forgot it entirely. would i go back? probably not, but maybe if they fix the noodles. prices are fine though, like 14 bucks for a big bowl.";

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export default function InputCard({
  text,
  setText,
  tab,
  setTab,
  file,
  setFile,
  onDetect,
  onFileDetect,
  onHumanize,
  loading,
  humanizing,
  modelReady,
  tone,
  setTone,
  strength,
  setStrength,
  target,
  setTarget,
}) {
  const fileInputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const words = countWords(text);
  const chars = text.length;
  const overLimit = chars > ANALYSIS_CHAR_LIMIT;
  const busy = loading || humanizing;

  function pickFiles(list) {
    const picked = list?.[0];
    if (picked) setFile(picked);
  }

  function run() {
    if (tab === "file") onFileDetect();
    else if (tab === "humanize") onHumanize();
    else onDetect();
  }

  const actionLabel =
    tab === "humanize"
      ? humanizing
        ? "Rewriting…"
        : "Make it AI-free"
      : loading
        ? "Analyzing…"
        : "Detect AI";

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-xl shadow-black/20 sm:p-5">
      <div className="mb-4 flex gap-1 rounded-xl bg-slate-950/70 p-1 ring-1 ring-slate-800">
        {[
          { id: "text", label: "Paste text" },
          { id: "file", label: "Upload file" },
          { id: "humanize", label: "AI-free" },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition ${
              tab === t.id
                ? "bg-slate-800 text-white shadow"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab !== "file" ? (
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
            placeholder={
              tab === "humanize"
                ? "Paste AI text you want to rewrite so it reads human…"
                : "Paste or type the text you want to check… (essays, emails, articles, posts)"
            }
            className="min-h-56 w-full resize-y rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-[15px] leading-relaxed text-slate-100 placeholder:text-slate-600 focus:border-sky-500/60 focus:outline-none focus:ring-2 focus:ring-sky-500/20"
          />
          <div className="pointer-events-none absolute bottom-3 right-4 flex gap-3 text-[11px] tabular-nums text-slate-500">
            <span>{words} words</span>
            <span>{chars.toLocaleString()} chars</span>
          </div>
        </div>
      ) : (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            pickFiles(e.dataTransfer.files);
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-6 text-center transition ${
            dragging
              ? "border-sky-500 bg-sky-500/5"
              : "border-slate-700 bg-slate-950/50 hover:border-slate-500"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.pdf,.docx"
            className="hidden"
            onChange={(e) => pickFiles(e.target.files)}
          />
          <span className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-800 ring-1 ring-slate-700">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-6 w-6 text-slate-400">
              <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16" strokeLinecap="round" />
            </svg>
          </span>
          {file ? (
            <div className="space-y-1">
              <p className="text-sm font-medium text-slate-200">{file.name}</p>
              <p className="text-xs text-slate-500">
                {(file.size / 1024).toFixed(0)} KB — click Analyze to scan
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              <p className="text-sm font-medium text-slate-300">
                Drop a file here or click to browse
              </p>
              <p className="text-xs text-slate-500">TXT, MD, PDF or DOCX — up to 10 MB</p>
            </div>
          )}
        </div>
      )}

      {tab === "text" && overLimit && (
        <p className="mt-3 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-300 ring-1 ring-inset ring-amber-500/20">
          Detection analyzes the first {ANALYSIS_CHAR_LIMIT.toLocaleString()} characters.
          Longer input is still fully rewritten on the AI-free tab.
        </p>
      )}

      {tab === "humanize" && (
        <div className="mt-3 space-y-3 rounded-xl bg-slate-950/60 px-3 py-3 ring-1 ring-slate-800">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Target
              </span>
              <div className="flex rounded-lg bg-slate-900 p-0.5 ring-1 ring-slate-800">
                {TARGETS.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setTarget(t.id)}
                    disabled={busy}
                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition disabled:opacity-50 ${
                      target === t.id
                        ? "bg-slate-700 text-white shadow"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>
            <p className="text-[11px] leading-relaxed text-slate-500 sm:text-right">
              {target === "external"
                ? "Ranks rewrites for ZeroGPT, GPTZero & Pangram proxies; layout preserved."
                : "Ranks rewrites for this detector only."}
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Tone
              </span>
              <div className="flex rounded-lg bg-slate-900 p-0.5 ring-1 ring-slate-800">
                {TONES.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setTone(t.id)}
                    disabled={busy}
                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition disabled:opacity-50 ${
                      tone === t.id
                        ? "bg-slate-700 text-white shadow"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Strength
              </span>
              <div className="flex rounded-lg bg-slate-900 p-0.5 ring-1 ring-slate-800">
                {STRENGTHS.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setStrength(s.id)}
                    disabled={busy}
                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition disabled:opacity-50 ${
                      strength === s.id
                        ? "bg-slate-700 text-white shadow"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex gap-2">
          <button
            onClick={() => setText(AI_EXAMPLE)}
            disabled={busy}
            className="rounded-lg px-3 py-2 text-xs font-medium text-slate-400 ring-1 ring-slate-800 transition hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
          >
            Try AI example
          </button>
          <button
            onClick={() => setText(HUMAN_EXAMPLE)}
            disabled={busy}
            className="rounded-lg px-3 py-2 text-xs font-medium text-slate-400 ring-1 ring-slate-800 transition hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
          >
            Try human example
          </button>
        </div>

        <button
          onClick={run}
          disabled={busy || !modelReady || (tab === "file" ? !file : !text.trim())}
          className={`inline-flex items-center justify-center gap-2 rounded-xl px-7 py-3 text-sm font-semibold text-white shadow-lg transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45 disabled:shadow-none ${
            tab === "humanize"
              ? "bg-gradient-to-r from-emerald-500 to-teal-500 shadow-emerald-500/25"
              : "bg-gradient-to-r from-sky-500 to-indigo-500 shadow-indigo-500/25"
          }`}
        >
          {busy ? <Spinner /> : null}
          {actionLabel}
        </button>
      </div>

      {!modelReady && (
        <p className="mt-3 flex items-center justify-center gap-2 text-xs text-slate-500">
          <Spinner />
          First-time model load in progress (~1 minute) — actions will unlock shortly.
        </p>
      )}
    </section>
  );
}
