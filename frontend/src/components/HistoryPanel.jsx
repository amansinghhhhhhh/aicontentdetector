import { useState } from "react";
import { verdictMeta } from "../lib/styles.js";

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HistoryPanel({ items, total, onLoad, onRefresh, activeId }) {
  const [busy, setBusy] = useState(false);

  async function clearAll() {
    if (!items.length) return;
    if (!window.confirm("Delete all scan history?")) return;
    setBusy(true);
    try {
      await fetch("/api/history", { method: "DELETE" });
      await onRefresh();
    } finally {
      setBusy(false);
    }
  }

  async function removeItem(e, id) {
    e.stopPropagation();
    setBusy(true);
    try {
      await fetch(`/api/history/${id}`, { method: "DELETE" });
      await onRefresh();
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-xl shadow-black/20">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200">
          History{" "}
          <span className="ml-1 text-xs font-normal text-slate-500">({total})</span>
        </h2>
        <button
          onClick={clearAll}
          disabled={busy || !items.length}
          className="text-[11px] text-slate-500 transition hover:text-rose-400 disabled:opacity-40"
        >
          Clear all
        </button>
      </div>

      {items.length === 0 ? (
        <p className="rounded-xl border border-dashed border-slate-800 px-3 py-6 text-center text-xs leading-relaxed text-slate-600">
          No scans yet.
          <br />
          Results are saved here automatically.
        </p>
      ) : (
        <ul className="max-h-[420px] space-y-2 overflow-y-auto pr-1">
          {items.map((item) => {
            const meta = verdictMeta(item.verdict);
            return (
              <li
                key={item.id}
                onClick={() => onLoad(item.id)}
                className={`group cursor-pointer rounded-xl border p-3 transition ${
                  activeId === item.id
                    ? "border-sky-500/50 bg-slate-800/70"
                    : "border-slate-800 bg-slate-950/50 hover:border-slate-700 hover:bg-slate-800/50"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2">
                    <i className={`h-2 w-2 shrink-0 rounded-full ${meta.bg}`} />
                    <span className="text-sm font-bold tabular-nums">
                      {item.ai_percent.toFixed(0)}%
                    </span>
                    <span className={`text-[11px] ${meta.text}`}>{meta.short}</span>
                  </span>
                  <button
                    onClick={(e) => removeItem(e, item.id)}
                    aria-label="Delete scan"
                    className="rounded p-1 text-slate-600 opacity-0 transition hover:bg-slate-700 hover:text-rose-400 group-hover:opacity-100"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-3.5 w-3.5">
                      <path d="M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                </div>
                <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-slate-500">
                  {item.filename ? `📄 ${item.filename} — ` : ""}
                  {item.preview}
                </p>
                <p className="mt-1.5 text-[10px] tabular-nums text-slate-600">
                  {formatDate(item.created_at)} · {item.words} words
                </p>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
