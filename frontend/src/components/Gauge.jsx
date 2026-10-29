import { useEffect, useState } from "react";
import { verdictMeta } from "../lib/styles.js";

function useCountUp(target, duration = 900) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let raf;
    const start = performance.now();
    const tick = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setValue(target * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);

  return value;
}

export default function Gauge({ percent, verdict, confidence }) {
  const meta = verdictMeta(verdict);
  const animated = useCountUp(percent);
  const radius = 86;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - animated / 100);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative h-52 w-52">
        <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke="#1e293b"
            strokeWidth="14"
          />
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke={meta.ring}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 900ms cubic-bezier(0.22,1,0.36,1)" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-extrabold tabular-nums tracking-tight">
            {animated.toFixed(0)}
            <span className="text-2xl font-bold text-slate-400">%</span>
          </span>
          <span className="mt-1 text-[11px] uppercase tracking-widest text-slate-500">
            AI content
          </span>
        </div>
      </div>
      <div className="flex flex-col items-center gap-1.5">
        <span
          className={`rounded-full px-3 py-1 text-sm font-semibold ring-1 ${meta.chip}`}
        >
          {meta.label}
        </span>
        <span className="text-xs text-slate-500">
          confidence {confidence?.toFixed?.(0) ?? confidence}%
        </span>
      </div>
    </div>
  );
}
