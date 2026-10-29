import { useCallback, useEffect, useState } from "react";
import Header from "./components/Header.jsx";
import InputCard from "./components/InputCard.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import HumanizePanel from "./components/HumanizePanel.jsx";
import HistoryPanel from "./components/HistoryPanel.jsx";
import { api } from "./lib/api.js";

export default function App() {
  const [text, setText] = useState("");
  const [tab, setTab] = useState("text");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [history, setHistory] = useState({ items: [], total: 0 });
  const [tone, setTone] = useState("casual");
  const [strength, setStrength] = useState("medium");
  const [target, setTarget] = useState("external");
  const [humanizeResult, setHumanizeResult] = useState(null);
  const [humanizing, setHumanizing] = useState(false);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await api.history());
    } catch {
      /* history refresh is best-effort */
    }
  }, []);

  useEffect(() => {
    let timer;
    let cancelled = false;

    const poll = async () => {
      try {
        const health = await api.health();
        if (cancelled) return;
        setModelReady(Boolean(health.model_loaded));
        if (!health.model_loaded) timer = setTimeout(poll, 3000);
      } catch {
        if (!cancelled) timer = setTimeout(poll, 5000);
      }
    };

    poll();
    refreshHistory();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [refreshHistory]);

  async function detectText(value) {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.detect(value));
      setHumanizeResult(null);
      await refreshHistory();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runDetect() {
    await detectText(text);
  }

  async function runFileDetect() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await api.detectFile(file));
      setHumanizeResult(null);
      await refreshHistory();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runHumanize() {
    setHumanizing(true);
    setError(null);
    try {
      setHumanizeResult(await api.humanize(text, tone, strength, target));
      setResult(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setHumanizing(false);
    }
  }

  async function loadScan(id) {
    setError(null);
    try {
      setResult(await api.historyItem(id));
      setHumanizeResult(null);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="min-h-screen">
      <Header />

      <main className="mx-auto max-w-6xl px-4 pb-16 pt-10 sm:px-6">
        <div className="mb-8 text-center">
          <h2 className="mx-auto max-w-2xl text-3xl font-extrabold tracking-tight sm:text-4xl">
            Is this{" "}
            <span className="bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
              AI-generated
            </span>{" "}
            or human?
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-sm leading-relaxed text-slate-400">
            Paste text or drop a document — get an AI percentage, a verdict, and
            sentence-level highlighting. Everything runs on your machine.
          </p>
        </div>

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="min-w-0 space-y-5">
            <InputCard
              text={text}
              setText={setText}
              tab={tab}
              setTab={setTab}
              file={file}
              setFile={setFile}
              onDetect={runDetect}
              onFileDetect={runFileDetect}
              onHumanize={runHumanize}
              loading={loading}
              humanizing={humanizing}
              modelReady={modelReady}
              tone={tone}
              setTone={setTone}
              strength={strength}
              setStrength={setStrength}
              target={target}
              setTarget={setTarget}
            />

            {error && (
              <div className="flex items-start justify-between gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                <span>{error}</span>
                <button
                  onClick={() => setError(null)}
                  className="shrink-0 text-rose-400 hover:text-rose-200"
                  aria-label="Dismiss"
                >
                  ✕
                </button>
              </div>
            )}

            <HumanizePanel
              result={humanizeResult}
              onUse={(value) => setText(value)}
              onCheck={(value) => {
                setText(value);
                detectText(value);
              }}
              checking={loading}
            />

            <ResultPanel result={result} />

            {!result && !humanizeResult && !error && (
              <div className="rounded-2xl border border-dashed border-slate-800 px-6 py-10 text-center">
                <p className="text-sm text-slate-500">
                  Your analysis will appear here — overall AI %, verdict, and each
                  sentence colour-coded by how AI-like it reads.
                </p>
              </div>
            )}
          </div>

          <HistoryPanel
            items={history.items}
            total={history.total}
            onLoad={loadScan}
            onRefresh={refreshHistory}
            activeId={result?.id}
          />
        </div>

        <footer className="mt-10 text-center text-[11px] leading-relaxed text-slate-600">
          Scores are evidence, not proof — keep a human in the loop.
          <br />
          Powered by an open-weights DeBERTa detector (Apache-2.0) running locally via FastAPI.
        </footer>
      </main>
    </div>
  );
}
