export function verdictMeta(verdict) {
  switch (verdict) {
    case "ai":
      return {
        label: "Likely AI-Generated",
        short: "AI",
        text: "text-rose-300",
        bg: "bg-rose-500",
        chip: "bg-rose-500/15 text-rose-300 ring-rose-500/30",
        ring: "#fb7185",
      };
    case "mixed":
      return {
        label: "Mixed / AI-Assisted",
        short: "Mixed",
        text: "text-amber-300",
        bg: "bg-amber-500",
        chip: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
        ring: "#fbbf24",
      };
    default:
      return {
        label: "Likely Human-Written",
        short: "Human",
        text: "text-emerald-300",
        bg: "bg-emerald-500",
        chip: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
        ring: "#34d399",
      };
  }
}

export function sentenceClass(score) {
  if (score >= 60) return "bg-rose-500/15 ring-1 ring-inset ring-rose-500/30";
  if (score >= 25) return "bg-amber-500/15 ring-1 ring-inset ring-amber-500/30";
  return "bg-emerald-500/10 ring-1 ring-inset ring-emerald-500/20";
}

export function countWords(text) {
  const trimmed = (text || "").trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}
