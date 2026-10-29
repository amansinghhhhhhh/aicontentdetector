export default function Header() {
  return (
    <header className="border-b border-slate-800/70 bg-slate-950/80 backdrop-blur sticky top-0 z-20">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-sky-400 to-indigo-500 shadow-lg shadow-indigo-500/20">
            <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5 text-white" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6l2.1 2.1m0-12.8l-2.1 2.1M7.7 16.3l-2.1 2.1" />
              <circle cx="12" cy="12" r="3.4" />
            </svg>
          </span>
          <div>
            <h1 className="text-[15px] font-bold leading-tight tracking-tight">
              AI Content Detector
            </h1>
            <p className="text-[11px] leading-tight text-slate-500">
              ZeroGPT-style detection, runs 100% local
            </p>
          </div>
        </div>
        <div className="hidden items-center gap-2 text-xs text-slate-500 sm:flex">
          <span className="rounded-full bg-slate-900 px-2.5 py-1 ring-1 ring-slate-800">
            English
          </span>
          <span className="rounded-full bg-slate-900 px-2.5 py-1 ring-1 ring-slate-800">
            Privacy-first
          </span>
        </div>
      </div>
    </header>
  );
}
