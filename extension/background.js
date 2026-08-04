// SynthScan background service worker (Manifest V3).
// Creates a context-menu item and orchestrates scanning selected text on a page
// through the self-hosted SynthScan API.

const DEFAULT_API_URL = "http://localhost:8000";

async function getApiUrl() {
  const stored = await chrome.storage.sync.get("apiUrl");
  return (stored.apiUrl && stored.apiUrl.trim()) || DEFAULT_API_URL;
}

async function scanText(text, apiUrl) {
  const res = await fetch(`${apiUrl}/scan/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, backend: "roberta" }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`SynthScan server error (${res.status}): ${body}`);
  }
  return res.json();
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "synthscan-scan-selection",
    title: "Scan selection with SynthScan",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "synthscan-scan-selection") return;
  const text = (info.selectionText || "").trim();
  if (!text) return;
  if (!tab || tab.id == null) return;

  // Tell the content script to show a "scanning..." state.
  chrome.tabs.sendMessage(tab.id, { type: "synthscan", status: "loading" }).catch(() => {});

  try {
    const apiUrl = await getApiUrl();
    const result = await scanText(text, apiUrl);
    chrome.tabs.sendMessage(tab.id, { type: "synthscan", status: "result", result, selection: text })
      .catch(() => {});
  } catch (err) {
    const message =
      "Could not reach the SynthScan server. Make sure it's running (synthscan serve) " +
      `and the API URL is correct in the extension popup. (${err.message})`;
    chrome.tabs.sendMessage(tab.id, { type: "synthscan", status: "error", message })
      .catch(() => {});
  }
});

// Allow the popup (or anything) to scan text programmatically.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === "synthscan:scan") {
    (async () => {
      try {
        const apiUrl = await getApiUrl();
        const result = await scanText(msg.text, apiUrl);
        sendResponse({ ok: true, result });
      } catch (err) {
        sendResponse({ ok: false, error: String(err && err.message) });
      }
    })();
    return true; // keep message channel open for async response
  }
  return undefined;
});
