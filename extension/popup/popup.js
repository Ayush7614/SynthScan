// SynthScan popup - scan pasted text against the configured self-hosted server.

const DEFAULT_API_URL = "http://localhost:8000";

const apiUrlInput = document.getElementById("apiUrl");
const textInput = document.getElementById("text");
const scanBtn = document.getElementById("scanBtn");
const statusEl = document.getElementById("status");

function pct(v) {
  return `${Math.round((v || 0) * 100)}%`;
}

function esc(str) {
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}

async function init() {
  const stored = await chrome.storage.sync.get("apiUrl");
  apiUrlInput.value = stored.apiUrl || DEFAULT_API_URL;
}

function render(result, apiUrl) {
  const color = result.is_ai ? "#c0392b" : (result.ai_probability >= 0.4 ? "#e67e22" : "#27ae60");
  let html = `<div class="verdict" style="color:${color}">${esc(result.verdict)}</div>`;
  html += `<div style="color:#444;margin-top:4px">AI probability: ${pct(result.ai_probability)}</div>`;
  html += `<div style="color:#888;margin-top:2px">backend: ${esc(result.backend)} · server: ${esc(apiUrl)}</div>`;

  if (result.segments && result.segments.length) {
    html += "<div style='margin-top:10px'><strong>Segment breakdown</strong></div>";
    for (const s of result.segments) {
      const cls = s.ai_probability >= 0.7 ? "ai" : (s.ai_probability >= 0.4 ? "mix" : "hum");
      html += `<div class="seg ${cls}">[${pct(s.ai_probability)}] ${esc(s.text)}</div>`;
    }
  }
  statusEl.innerHTML = html;
}

scanBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  const apiUrl = (apiUrlInput.value.trim() || DEFAULT_API_URL).replace(/\/$/, "");
  await chrome.storage.sync.set({ apiUrl });
  if (!text) {
    statusEl.textContent = "Please enter some text to scan.";
    return;
  }

  scanBtn.disabled = true;
  statusEl.textContent = "Scanning…";

  try {
    const res = await fetch(`${apiUrl}/scan/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, backend: "roberta" }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Server responded with ${res.status}: ${body}`);
    }
    const result = await res.json();
    render(result, apiUrl);
  } catch (err) {
    statusEl.innerHTML =
      `<div style="color:#c0392b"><strong>Error</strong><br/>${esc(err.message)}</div>` +
      `<div style="color:#666;margin-top:6px">Is your SynthScan server running? Correct it in the URL field and try again.</div>`;
  } finally {
    scanBtn.disabled = false;
  }
});

init();
