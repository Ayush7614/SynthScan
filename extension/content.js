// SynthScan content script - renders an in-page overlay result when the context
// menu scan completes (or shows a loading/error state), and adds a small inline
// highlight of which segments were flagged as AI.

(function () {
  if (window.__synthscanInjected) return;
  window.__synthscanInjected = true;

  let overlay = null;

  function removeOverlay() {
    if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay);
    overlay = null;
  }

  function esc(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
  }

  function pct(v) {
    return `${Math.round((v || 0) * 100)}%`;
  }

  function showOverlay(state) {
    removeOverlay();

    overlay = document.createElement("div");
    overlay.style.cssText =
      "position:fixed;right:16px;bottom:16px;z-index:2147483647;max-width:380px;" +
      "width:90%;background:#fff;border:1px solid #ddd;border-radius:12px;" +
      "box-shadow:0 8px 30px rgba(0,0,0,.25);font-family:-apple-system,Segoe UI,Roboto,sans-serif;" +
      "color:#111;padding:16px;font-size:14px;";

    const close = document.createElement("button");
    close.textContent = "✕";
    close.style.cssText =
      "float:right;border:none;background:none;font-size:16px;cursor:pointer;color:#888;";
    close.addEventListener("click", removeOverlay);
    overlay.appendChild(close);

    if (state.status === "loading") {
      overlay.insertAdjacentHTML("beforeend",
        `<div><strong>SynthScan</strong> — scanning selected text…</div>
         <div style="margin-top:8px;color:#666">Detecting AI-generated content…</div>`);
    } else if (state.status === "error") {
      overlay.insertAdjacentHTML("beforeend",
        `<div><strong>SynthScan</strong> — error</div>
         <div style="margin-top:8px;color:#c0392b">${esc(state.message)}</div>`);
    } else if (state.status === "result") {
      const r = state.result;
      const color = r.is_ai ? "#c0392b" : (r.ai_probability >= 0.4 ? "#e67e22" : "#27ae60");
      const label = r.verdict;
      overlay.insertAdjacentHTML("beforeend",
        `<div><strong>SynthScan</strong></div>
         <div style="margin-top:4px;font-size:16px;font-weight:600;color:${color}">
           ${esc(label)}
         </div>
         <div style="color:#444;margin-top:2px">AI probability: ${pct(r.ai_probability)}</div>
         <div style="color:#888;margin-top:2px">backend: ${esc(r.backend)}</div>`);
    }

    document.body.appendChild(overlay);
    setTimeout(removeOverlay, 20000); // auto-dismiss
  }

  // Highlight AI segments inside the currently selected text container (best-effort).
  function highlightSegments(selectionText, result) {
    removeOverlay();
    if (!window.getSelection || !result || !result.segments) return;
    const range = window.getSelection().getRangeAt(0);
    if (!range || !range.commonAncestorContainer) return;

    let container = range.commonAncestorContainer;
    if (container.nodeType === Node.TEXT_NODE) container = container.parentNode;
    if (!container || typeof container.innerHTML !== "string") return;

    let html = container.innerHTML;
    const aiSegments = result.segments.filter((s) => s.ai_probability >= 0.4);
    if (!aiSegments.length) return;

    // Find each flagged segment in the HTML source and wrap in a highlight span.
    for (const seg of aiSegments) {
      const needle = seg.text
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 80);
      if (!needle) continue;
      // Escape regex special chars.
      const escaped = needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const re = new RegExp(escaped, "i");
      const match = html.match(re);
      if (match) {
        const hl = `<mark style="background:#fde68a;padding:0 2px;border-radius:3px;">${match[0]}</mark>`;
        html = html.replace(re, hl);
      }
    }
    container.innerHTML = html;
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg || msg.type !== "synthscan") return;
    if (msg.status === "result" && msg.selection) {
      highlightSegments(msg.selection, msg.result);
      showOverlay(msg);
    } else {
      showOverlay(msg);
    }
  });
})();
