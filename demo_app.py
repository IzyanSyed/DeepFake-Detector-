#!/usr/bin/env python3
"""
Deepfake Detector — local demo dashboard (editorial / Awwwards-leaning design).
 
Run this, it opens Chrome automatically, shows all 8 planned pipeline
modules. Only "Spatial Artifacts" is wired up to the real trained model;
the other 7 show a "not developed yet" placeholder.
 
Usage:
    python demo_app.py
    python demo_app.py --checkpoint checkpoints/spatial_artifacts.pt --port 5050
 
Requires: flask (pip install flask --break-system-packages if needed)
"""
 
import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import webbrowser
 
from flask import Flask, request, jsonify, render_template_string
 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
 
try:
    from deepfake_detector.detectors.spatial_artifacts import SpatialArtifactsDetector
    from deepfake_detector.ingestion.stream_reader import SignalWindow
except ImportError as e:
    print("ERROR: Could not import project modules.")
    print(f"Make sure this file sits next to 'src/deepfake_detector/' and the import works: {e}")
    sys.exit(1)
 
app = Flask(__name__)
 
DETECTOR = None
CHECKPOINT_PATH = None
 
MODULES = [
    {"id": "spatial", "name": "Spatial Artifacts", "num": "01",
     "tagline": "Pixel + frequency domain analysis",
     "description": "EfficientNet-B0 pixel branch and an FFT radial-energy frequency branch, fused into one score.",
     "implemented": True},
    {"id": "physiological", "name": "Physiological", "num": "02",
     "tagline": "Heartbeat signal from skin color",
     "description": "Extracts a pulse signal from micro skin-color changes caused by blood flow.",
     "implemented": False},
    {"id": "temporal", "name": "Temporal Consistency", "num": "03",
     "tagline": "Frame-to-frame flicker & pose drift",
     "description": "Detects unnatural head-pose interpolation and inter-frame flicker.",
     "implemented": False},
    {"id": "av_sync", "name": "Audio-Visual Sync", "num": "04",
     "tagline": "Phoneme-to-viseme alignment",
     "description": "Checks whether cloned audio matches the reenacted face's mouth movement.",
     "implemented": False},
    {"id": "provenance", "name": "Provenance", "num": "05",
     "tagline": "C2PA content credentials",
     "description": "Fast 'definitely real' signal when valid content credentials are present.",
     "implemented": False},
    {"id": "fusion", "name": "Fusion Model", "num": "06",
     "tagline": "Combines all detector scores",
     "description": "Weighted average now, swappable for a learned fusion model later.",
     "implemented": False},
    {"id": "policy", "name": "Alert Policy", "num": "07",
     "tagline": "Hysteresis-based alerting",
     "description": "Requires N consecutive low-score windows before firing an alert.",
     "implemented": False},
    {"id": "orchestrator", "name": "Live Orchestrator", "num": "08",
     "tagline": "Real-time pipeline on a live call",
     "description": "Runs all detectors concurrently on a live video/audio stream.",
     "implemented": False},
]
 
 
def get_detector():
    global DETECTOR
    if DETECTOR is None:
        print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
        DETECTOR = SpatialArtifactsDetector()
        if hasattr(DETECTOR, "load_checkpoint") and CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            DETECTOR.load_checkpoint(CHECKPOINT_PATH)
            print("Checkpoint loaded.")
        else:
            print("WARNING: no checkpoint loaded — using untrained/backbone-only weights.")
    return DETECTOR
 
 
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Deepfake Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,400&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0b;
    --bg-2: #111113;
    --panel: #151517;
    --line: rgba(255,255,255,0.09);
    --text: #f4f3ef;
    --muted: #8c8b89;
    --accent: #d8ff3e;
    --real: #6ee7a8;
    --fake: #ff6b6b;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
    min-height: 100vh;
  }
  .grain {
    position: fixed; inset: 0; z-index: 999; pointer-events: none; opacity: 0.035; mix-blend-mode: overlay;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }
 
  /* ---------------- Header / Hero ---------------- */
  header {
    padding: 28px 48px 0;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 13px; letter-spacing: 0.04em; color: var(--muted);
    animation: fadeDown 0.7s ease both;
  }
  header .mark { color: var(--text); font-weight: 600; letter-spacing: 0.01em; }
  header .status { display: flex; align-items: center; gap: 8px; }
  header .status .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--real); box-shadow: 0 0 8px rgba(110,231,168,0.7); }
 
  .hero {
    padding: 90px 48px 70px;
    border-bottom: 1px solid var(--line);
  }
  .hero .eyebrow {
    font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
    margin-bottom: 22px; font-weight: 600;
    animation: fadeUp 0.7s ease 0.05s both;
  }
  .hero h1 {
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: clamp(42px, 6.2vw, 92px);
    line-height: 0.98;
    letter-spacing: -0.02em;
    margin: 0 0 26px;
    max-width: 920px;
    animation: fadeUp 0.8s ease 0.1s both;
  }
  .hero h1 em { font-style: italic; color: var(--accent); font-weight: 400; }
  .hero p.sub {
    font-size: 17px; color: var(--muted); max-width: 540px; line-height: 1.6; margin: 0;
    animation: fadeUp 0.8s ease 0.2s both;
  }
 
  @keyframes fadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes fadeDown { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
 
  /* ---------------- Module grid ---------------- */
  .grid-section { padding: 70px 48px 100px; }
  .grid-section .label {
    font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);
    margin-bottom: 28px; display: flex; align-items: center; gap: 14px;
  }
  .grid-section .label::after { content: ""; flex: 1; height: 1px; background: var(--line); }
 
  .modgrid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--line);
    border: 1px solid var(--line);
  }
  .modcard {
    background: var(--bg);
    padding: 30px 26px;
    cursor: pointer;
    transition: background 0.2s ease;
    position: relative;
    min-height: 190px;
    display: flex; flex-direction: column; justify-content: space-between;
  }
  .modcard:hover { background: var(--bg-2); }
  .modcard.active { background: var(--panel); }
  .modcard.active::after {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent);
  }
  .modcard .num { font-size: 12px; color: var(--muted); font-weight: 500; }
  .modcard .name { font-family: 'Fraunces', serif; font-size: 21px; font-weight: 500; margin: 14px 0 6px; line-height: 1.15; }
  .modcard .tag { font-size: 12.5px; color: var(--muted); line-height: 1.4; }
  .modcard .state {
    margin-top: 16px; font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase;
    display: inline-flex; align-items: center; gap: 6px; width: fit-content;
    padding: 5px 10px; border-radius: 999px; border: 1px solid var(--line);
  }
  .modcard .state .dot { width: 5px; height: 5px; border-radius: 50%; }
  .modcard .state.live { color: var(--real); border-color: rgba(110,231,168,0.3); }
  .modcard .state.live .dot { background: var(--real); }
  .modcard .state.pending { color: var(--muted); }
  .modcard .state.pending .dot { background: #4a4a4e; }
 
  @media (max-width: 900px) { .modgrid { grid-template-columns: repeat(2, 1fr); } }
 
  /* ---------------- Detail panel ---------------- */
  .detail { padding: 0 48px 110px; max-width: 760px; }
  .detail-head { margin-bottom: 34px; }
  .detail-head .eyebrow { font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); margin-bottom: 14px; font-weight: 600; }
  .detail-head h2 { font-family: 'Fraunces', serif; font-size: clamp(32px, 4vw, 48px); font-weight: 500; margin: 0 0 14px; letter-spacing: -0.01em; }
  .detail-head p { color: var(--muted); font-size: 15.5px; line-height: 1.6; max-width: 540px; margin: 0; }
 
  .placeholder {
    border: 1px solid var(--line); padding: 80px 40px; text-align: center; background: var(--bg-2);
  }
  .placeholder .ph-num { font-family: 'Fraunces', serif; font-style: italic; font-size: 54px; color: var(--line); margin-bottom: 18px; }
  .placeholder h3 { font-family: 'Fraunces', serif; font-size: 22px; font-weight: 500; margin: 0 0 10px; }
  .placeholder p { color: var(--muted); font-size: 13.5px; max-width: 360px; margin: 0 auto; line-height: 1.6; }
 
  .card { background: var(--bg-2); border: 1px solid var(--line); padding: 34px; }
  #dropzone {
    border: 1px dashed var(--line); padding: 44px 20px; text-align: center; cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease;
  }
  #dropzone.drag { border-color: var(--accent); background: rgba(216,255,62,0.04); }
  #dropzone .main-text { color: var(--text); font-size: 15px; font-weight: 500; margin: 10px 0 4px; }
  #dropzone .sub-text { color: var(--muted); font-size: 12.5px; }
  input[type=file] { display: none; }
  #filename { margin-top: 16px; font-size: 13px; color: var(--muted); text-align: center; word-break: break-all; }
 
  .run-btn {
    margin-top: 22px; width: 100%; padding: 15px; border: none;
    background: var(--accent); color: #0a0a0b; font-size: 13.5px; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase; cursor: pointer;
    transition: opacity 0.15s ease, transform 0.1s ease;
  }
  .run-btn:disabled { opacity: 0.25; cursor: not-allowed; }
  .run-btn:not(:disabled):hover { opacity: 0.88; }
  .run-btn:not(:disabled):active { transform: scale(0.99); }
 
  #status { text-align: center; margin-top: 16px; color: var(--muted); font-size: 13px; min-height: 18px; }
 
  .result { margin-top: 30px; border-top: 1px solid var(--line); padding-top: 30px; display: none; }
  .result.show { display: block; animation: fadeUp 0.5s ease both; }
  .decision-row { display: flex; justify-content: center; margin-bottom: 24px; }
  .decision-badge {
    font-family: 'Fraunces', serif; font-size: 34px; font-weight: 500; letter-spacing: -0.01em;
    padding: 6px 0;
  }
  .decision-badge.real { color: var(--real); }
  .decision-badge.fake { color: var(--fake); }
 
  .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); border: 1px solid var(--line); }
  .metric { background: var(--bg); padding: 18px; }
  .metric .label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
  .metric .value { font-family: 'Fraunces', serif; font-size: 24px; font-weight: 500; margin-top: 6px; }
 
  .bar-wrap { margin-top: 22px; }
  .bar-label { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--muted); margin-bottom: 8px; letter-spacing: 0.03em; }
  .bar-track { height: 3px; background: var(--line); overflow: hidden; }
  .bar-fill { height: 100%; transition: width 0.5s ease; }
 
  .disclaimer { margin-top: 22px; font-size: 11px; color: var(--muted); line-height: 1.6; }
  .raw-toggle { margin-top: 16px; }
  .raw-toggle a { color: var(--accent); font-size: 11.5px; text-decoration: none; cursor: pointer; letter-spacing: 0.02em; }
  pre#raw { display: none; margin-top: 12px; background: var(--bg); border: 1px solid var(--line); padding: 14px; font-size: 11px; color: var(--muted); overflow-x: auto; }
  pre#raw.show { display: block; }
 
  footer { padding: 30px 48px; border-top: 1px solid var(--line); font-size: 11.5px; color: var(--muted); display: flex; justify-content: space-between; }
</style>
</head>
<body>
<div class="grain"></div>
 
<header>
  <div class="mark">DEEPFAKE DETECTOR</div>
  <div class="status"><div class="dot"></div> SYSTEM ONLINE</div>
</header>
 
<div class="hero">
  <div class="eyebrow">Real-time multimodal detection</div>
  <h1>Proving someone is <em>human</em> — before the wire transfer clears.</h1>
  <p class="sub">An eight-module pipeline fusing spatial, physiological, temporal, audio and provenance signals into one real-time verdict.</p>
</div>
 
<div class="grid-section">
  <div class="label">Pipeline Modules</div>
  <div class="modgrid" id="modgrid"></div>
</div>
 
<div class="detail" id="detail"></div>
 
<footer>
  <span>Built on real DFDC competition data</span>
  <span id="footerModule">Module 01 / 08</span>
</footer>
 
<script>
  const MODULES = __MODULES_JSON__;
  let currentId = MODULES.find(m => m.implemented).id;
 
  function renderGrid() {
    const grid = document.getElementById('modgrid');
    grid.innerHTML = '';
    MODULES.forEach((m, i) => {
      const card = document.createElement('div');
      card.className = 'modcard' + (m.id === currentId ? ' active' : '');
      card.style.animation = `fadeUp 0.6s ease ${0.05 * i}s both`;
      card.innerHTML = `
        <div class="num">${m.num}</div>
        <div>
          <div class="name">${m.name}</div>
          <div class="tag">${m.tagline}</div>
        </div>
        <div class="state ${m.implemented ? 'live' : 'pending'}">
          <div class="dot"></div>${m.implemented ? 'Live' : 'Planned'}
        </div>
      `;
      card.addEventListener('click', () => {
        currentId = m.id;
        renderGrid();
        renderDetail();
        document.getElementById('footerModule').textContent = `Module ${m.num} / 08`;
        document.getElementById('detail').scrollIntoView({behavior: 'smooth', block: 'start'});
      });
      grid.appendChild(card);
    });
  }
 
  function renderDetail() {
    const m = MODULES.find(x => x.id === currentId);
    const detail = document.getElementById('detail');
 
    if (!m.implemented) {
      detail.innerHTML = `
        <div class="detail-head">
          <div class="eyebrow">Module ${m.num}</div>
          <h2>${m.name}</h2>
          <p>${m.description}</p>
        </div>
        <div class="placeholder">
          <div class="ph-num">${m.num}</div>
          <h3>Not developed yet</h3>
          <p>This module is planned but hasn't been built. Only Spatial Artifacts is currently trained and wired up end-to-end.</p>
        </div>
      `;
      return;
    }
 
    detail.innerHTML = `
      <div class="detail-head">
        <div class="eyebrow">Module ${m.num} — Live</div>
        <h2>${m.name}</h2>
        <p>${m.description}</p>
      </div>
      <div class="card">
        <div id="dropzone">
          <div class="main-text">Drop a video here, or click to browse</div>
          <div class="sub-text">.mp4 · .mov · .avi</div>
        </div>
        <input type="file" id="fileInput" accept="video/*" />
        <div id="filename"></div>
        <button class="run-btn" id="runBtn" disabled>Analyze Video</button>
        <div id="status"></div>
        <div class="result" id="result">
          <div class="decision-row"><div class="decision-badge" id="decisionBadge">—</div></div>
          <div class="metrics">
            <div class="metric"><div class="label">Fake Score</div><div class="value" id="scoreVal">—</div></div>
            <div class="metric"><div class="label">Latency</div><div class="value" id="latencyVal">—</div></div>
          </div>
          <div class="bar-wrap">
            <div class="bar-label"><span>CONFIDENCE</span><span id="confLabel">—</span></div>
            <div class="bar-track"><div class="bar-fill" id="confBar" style="width:0%"></div></div>
          </div>
          <div class="raw-toggle"><a id="rawToggle">Show raw signals →</a></div>
          <pre id="raw"></pre>
          <div class="disclaimer">Confidence is a heuristic (distance from the 0.5 decision boundary), not a calibrated probability. Model trained on a small sample for demo purposes.</div>
        </div>
      </div>
    `;
    wireUploader();
  }
 
  function wireUploader() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const filenameEl = document.getElementById('filename');
    const runBtn = document.getElementById('runBtn');
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    const rawToggle = document.getElementById('rawToggle');
    const rawEl = document.getElementById('raw');
    let selectedFile = null;
 
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault(); dropzone.classList.remove('drag');
      if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => { if (e.target.files.length) setFile(e.target.files[0]); });
 
    function setFile(file) {
      selectedFile = file;
      filenameEl.textContent = file.name;
      runBtn.disabled = false;
      resultEl.classList.remove('show');
      statusEl.textContent = '';
    }
 
    rawToggle.addEventListener('click', () => rawEl.classList.toggle('show'));
 
    runBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      runBtn.disabled = true;
      statusEl.textContent = 'Analyzing — this can take a few seconds';
      resultEl.classList.remove('show');
 
      const formData = new FormData();
      formData.append('video', selectedFile);
 
      try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) {
          statusEl.textContent = 'Error: ' + data.error;
          runBtn.disabled = false;
          return;
        }
        renderResult(data);
        statusEl.textContent = '';
      } catch (err) {
        statusEl.textContent = 'Request failed: ' + err;
      }
      runBtn.disabled = false;
    });
 
    function renderResult(data) {
      const badge = document.getElementById('decisionBadge');
      badge.textContent = data.decision;
      badge.className = 'decision-badge ' + (data.decision === 'FAKE' ? 'fake' : 'real');
      document.getElementById('scoreVal').textContent = data.score.toFixed(3);
      document.getElementById('latencyVal').textContent = Math.round(data.latency_ms) + 'ms';
      const confPct = data.confidence_pct.toFixed(1);
      document.getElementById('confLabel').textContent = confPct + '%';
      const bar = document.getElementById('confBar');
      bar.style.width = confPct + '%';
      bar.style.background = data.decision === 'FAKE' ? 'var(--fake)' : 'var(--real)';
      rawEl.textContent = JSON.stringify(data.raw_signals || {}, null, 2);
      resultEl.classList.add('show');
    }
  }
 
  renderGrid();
  renderDetail();
</script>
</body>
</html>
"""
 
 
@app.route("/")
def index():
    html = PAGE.replace("__MODULES_JSON__", json.dumps(MODULES))
    return render_template_string(html)
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "no video uploaded"}), 400
 
    video_file = request.files["video"]
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
 
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name
 
    try:
        detector = get_detector()
 
        if hasattr(detector, "analyze_video_file"):
            result = detector.analyze_video_file(tmp_path)
        else:
            import cv2
            import numpy as np
 
            cap = cv2.VideoCapture(tmp_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idxs = np.linspace(0, max(frame_count - 1, 0), min(5, max(frame_count, 1)), dtype=int)
            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
 
            window = SignalWindow(frames=frames, timestamp=time.time(), window_id="demo-ui")
            result = asyncio.run(detector.analyze(window))
 
        score = float(result.score)
        decision = "FAKE" if score > 0.5 else "REAL"
        confidence_pct = min(abs(score - 0.5) * 200, 100.0)
 
        return jsonify({
            "decision": decision,
            "score": score,
            "confidence_pct": confidence_pct,
            "latency_ms": float(result.latency_ms),
            "raw_signals": result.raw_signals,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
 
 
def open_browser(url):
    time.sleep(1.2)
    try:
        chrome = webbrowser.get("chrome")
        chrome.open(url)
    except webbrowser.Error:
        webbrowser.open(url)
 
 
def main():
    global CHECKPOINT_PATH
    parser = argparse.ArgumentParser(description="Deepfake Detector local demo dashboard")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/spatial_artifacts.pt")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
 
    CHECKPOINT_PATH = args.checkpoint
    url = f"http://127.0.0.1:{args.port}"
 
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
 
    print(f"\nStarting Deepfake Detector dashboard at {url}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=args.port, debug=False)
 
 
if __name__ == "__main__":
    main()
 