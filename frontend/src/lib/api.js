async function request(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    let detail = data.detail;
    if (Array.isArray(detail)) {
      detail = detail.map((d) => d.msg).join("; ");
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

export const api = {
  health: () => request("/api/health"),

  detect: (text) =>
    request("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }),

  detectFile: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/detect/file", { method: "POST", body: form });
  },

  humanize: (text, tone, strength, target = "external") =>
    request("/api/humanize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, tone, strength, target }),
    }),

  history: () => request("/api/history"),

  historyItem: (id) => request(`/api/history/${id}`),

  deleteHistoryItem: (id) => request(`/api/history/${id}`, { method: "DELETE" }),

  clearHistory: () => request("/api/history", { method: "DELETE" }),
};
