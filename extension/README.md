# SynthScan Browser Extension

Detect AI-generated text on any webpage using your **self-hosted** SynthScan server.

## What it does
- **Context menu:** select text on any page → right-click → **"Scan selection with SynthScan"** → a result overlay appears with the verdict, AI probability, and highlights of flagged segments.
- **Popup:** click the SynthScan toolbar icon to open a panel where you can paste any text to scan, and configure the server URL.

## Requirements
- A running SynthScan server (the extension is a *client* for it):
  ```bash
  pip install -e ".[api,ml]"          # once
  synthscan serve --host 0.0.0.0 --port 8001
  ```
  > The extension defaults to the `roberta` backend for accurate, CPU-friendly results. Use the `heuristic` backend if you haven't installed the ML extras yet.
  >
  > **Why port 8001?** Port 8000 is commonly occupied by other software (it was on our machine), so the extension defaults to 8001. You can use any port — just set the server URL in the popup to match.

## Install in Chrome / Edge (load unpacked)
1. Open `chrome://extensions` (or `edge://extensions`).
2. Enable **Developer mode** (top-right).
3. Click **Load unpacked** and select this `extension/` folder.
4. The SynthScan icon appears in the toolbar.

## Configure the server URL
- The default is `http://localhost:8001`. If your SynthScan server runs elsewhere (e.g. a LAN IP or a public instance), open the extension popup and update the **SynthScan server URL** field. It is saved automatically.

## Permissions (why)
- `activeTab` + `contextMenus` — for the "Scan selection" right-click action.
- `storage` — to remember your server URL.
- Host permission for `localhost:8000` / `localhost:8001` — the default servers. (Extend `host_permissions` in `manifest.json` if you use a different origin.)

## Privacy
The extension sends scanned text **only to the SynthScan server URL you configure** — by default your own machine. It never phones home to any third party.

## Reproduce the icons
Icons are generated (no binaries in git) with:
```bash
python scripts/generate_icons.py
```

