// ---- System status ----

async function loadHealth() {
  const list = document.getElementById("status-list");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    list.innerHTML = "";

    list.appendChild(statusRow("ffmpeg", data.ffmpeg.found, data.ffmpeg.version || data.ffmpeg.error));

    const gpuDetail = data.gpu.found
      ? `${data.gpu.name} (${(data.gpu.vram_total_mb / 1024).toFixed(1)} GB VRAM, driver ${data.gpu.driver_version})`
      : data.gpu.error;
    list.appendChild(statusRow("GPU", data.gpu.found, gpuDetail, data.gpu.found ? "ok" : "warn"));

    for (const disk of data.disk) {
      const state = disk.low_space_warning ? "warn" : "ok";
      list.appendChild(statusRow(`disk (${disk.path})`, true, `${disk.free_gb} GB free of ${disk.total_gb} GB`, state));
    }
  } catch (e) {
    list.innerHTML = `<div class="status-row"><span class="dot err"></span>Could not reach server: ${e}</div>`;
  }
}

function statusRow(label, ok, detail, forceState) {
  const div = document.createElement("div");
  div.className = "status-row";
  const state = forceState || (ok ? "ok" : "err");
  div.innerHTML = `<span><span class="dot ${state}"></span>${label}</span><span class="detail">${detail || ""}</span>`;
  return div;
}

// ---- State ----

const state = {
  uploadedPath: null,
  jobId: null,
  presets: {},
  batchMode: false,
  batchFiles: [], // {path, filename}
  batchId: null,
};

// ---- Presets ----

async function loadPresets() {
  const select = document.getElementById("preset-select");
  if (!select) {
    console.error("loadPresets: #preset-select not found in the DOM (stale page cache?)");
    return;
  }
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.presets = await res.json();
    select.innerHTML = "";
    for (const [key, preset] of Object.entries(state.presets)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = preset.name;
      select.appendChild(opt);
    }
    updatePresetDescription();
  } catch (e) {
    console.error("Failed to load presets:", e);
    select.innerHTML = '<option value="">Failed to load presets - reload the page</option>';
  }
}

function updatePresetDescription() {
  const select = document.getElementById("preset-select");
  const desc = document.getElementById("preset-description");
  const preset = state.presets[select.value];
  desc.textContent = preset ? preset.description : "";
}

// ---- Upload + analyze ----

function setupDropzone() {
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  if (!zone || !input) {
    console.error("setupDropzone: #dropzone or #file-input not found in the DOM (stale page cache?)");
    return;
  }

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) handleFiles(input.files);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
    })
  );
  zone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
  });
}

async function handleFiles(fileList) {
  resetForNewFile();
  const files = Array.from(fileList);
  if (files.length === 1) {
    state.batchMode = false;
    await handleSingleFile(files[0]);
  } else {
    state.batchMode = true;
    await handleBatchFiles(files);
  }
}

async function handleSingleFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const uploadRes = await fetch("/api/upload", { method: "POST", body: fd });
  if (!uploadRes.ok) {
    alert("Upload failed: " + (await uploadRes.text()));
    return;
  }
  const uploadData = await uploadRes.json();
  state.uploadedPath = uploadData.path;

  const analyzeRes = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: uploadData.path }),
  });
  if (!analyzeRes.ok) {
    alert("Analysis failed: " + (await analyzeRes.text()));
    return;
  }
  const analyzeData = await analyzeRes.json();
  renderAnalysis(uploadData.filename, analyzeData);
}

async function handleBatchFiles(files) {
  state.batchFiles = [];
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) {
      alert(`Upload failed for ${file.name}: ` + (await res.text()));
      continue;
    }
    const data = await res.json();
    state.batchFiles.push({ path: data.path, filename: data.filename });
  }
  renderBatchFileList();
  if (state.batchFiles.length) {
    document.getElementById("pipeline-panel").classList.remove("hidden");
    updatePresetDescription();
  }
}

function renderBatchFileList() {
  const container = document.getElementById("batch-file-list");
  if (!state.batchFiles.length) {
    container.classList.add("hidden");
    container.innerHTML = "";
    return;
  }
  container.classList.remove("hidden");
  container.innerHTML =
    `<p class="detail">${state.batchFiles.length} file(s) queued for batch processing:</p>` +
    state.batchFiles
      .map(
        (f, i) =>
          `<div class="file-list-row"><span>${f.filename}</span><button class="remove-file-btn" data-idx="${i}">remove</button></div>`
      )
      .join("");
  container.querySelectorAll(".remove-file-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const idx = Number(e.currentTarget.dataset.idx);
      state.batchFiles.splice(idx, 1);
      renderBatchFileList();
      if (!state.batchFiles.length) document.getElementById("pipeline-panel").classList.add("hidden");
    });
  });
}

function resetForNewFile() {
  document.getElementById("results-panel").classList.add("hidden");
  document.getElementById("progress-panel").classList.add("hidden");
  document.getElementById("batch-progress-panel").classList.add("hidden");
  document.getElementById("file-info").classList.add("hidden");
  document.getElementById("batch-file-list").classList.add("hidden");
  state.jobId = null;
  state.batchId = null;
  state.batchFiles = [];
}

function renderAnalysis(filename, data) {
  const a = data.analysis;
  document.getElementById("file-info").classList.remove("hidden");
  document.getElementById("file-name").textContent = filename;
  document.getElementById("file-duration").textContent = formatDuration(a.duration_seconds);

  const warningsDiv = document.getElementById("analysis-warnings");
  warningsDiv.innerHTML = "";
  for (const w of a.warnings) {
    const row = document.createElement("div");
    row.className = "warning-row";
    row.textContent = "⚠ " + w;
    warningsDiv.appendChild(row);
  }

  const metrics = document.getElementById("analysis-metrics");
  metrics.innerHTML = "";
  const items = [
    ["Peak", `${a.peak_dbfs.toFixed(1)} dBFS`],
    ["Loudness", `${a.lufs.toFixed(1)} LUFS`],
    ["Clipping", `${a.clipping_percent.toFixed(2)}%`],
    ["Noise floor", `${a.noise_floor_dbfs.toFixed(1)} dBFS`],
    ["Bandwidth", `${Math.round(a.bandwidth_hz)} Hz`],
    ["Mode", a.mode_guess],
  ];
  for (const [label, value] of items) {
    const div = document.createElement("div");
    div.className = "metric";
    div.innerHTML = `<span class="label">${label}</span>${value}`;
    metrics.appendChild(div);
  }

  document.getElementById("pipeline-panel").classList.remove("hidden");
  const select = document.getElementById("preset-select");
  if (state.presets[data.suggested_preset]) select.value = data.suggested_preset;
  updatePresetDescription();
}

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---- Run job + progress ----

function setupRunButton() {
  document.getElementById("run-button").addEventListener("click", () => {
    if (state.batchMode) runBatch();
    else runJob();
  });
  document.getElementById("preset-select").addEventListener("change", updatePresetDescription);
}

async function runJob() {
  if (!state.uploadedPath) return;
  const preset = document.getElementById("preset-select").value;

  document.getElementById("orig-audio").pause();
  document.getElementById("final-audio").pause();
  document.getElementById("play-button").textContent = "Play";

  document.getElementById("run-button").disabled = true;
  document.getElementById("results-panel").classList.add("hidden");
  const progressPanel = document.getElementById("progress-panel");
  progressPanel.classList.remove("hidden");
  setProgress("starting", 0);

  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: state.uploadedPath, preset }),
  });
  if (!res.ok) {
    alert("Failed to start job: " + (await res.text()));
    document.getElementById("run-button").disabled = false;
    return;
  }
  const { job_id } = await res.json();
  state.jobId = job_id;

  const es = new EventSource(`/api/jobs/${job_id}/events`);
  es.onmessage = (e) => {
    const item = JSON.parse(e.data);
    if (item.type === "progress") {
      setProgress(item.stage, item.progress);
    } else if (item.type === "done") {
      es.close();
      document.getElementById("run-button").disabled = false;
      loadResults(job_id);
      loadHistory();
    } else if (item.type === "error") {
      es.close();
      document.getElementById("run-button").disabled = false;
      alert("Job failed: " + item.error);
    }
  };
  es.onerror = () => {
    // EventSource retries automatically; nothing to do unless the job itself errored,
    // which is reported through a normal message above.
  };
}

function setProgress(stage, frac) {
  document.getElementById("stage-label").textContent = `${stage} - ${Math.round(frac * 100)}%`;
  document.getElementById("progress-fill").style.width = `${Math.round(frac * 100)}%`;
}

// ---- Batch queue ----

async function runBatch() {
  if (!state.batchFiles.length) return;
  const preset = document.getElementById("preset-select").value;

  document.getElementById("run-button").disabled = true;
  document.getElementById("results-panel").classList.add("hidden");
  const batchPanel = document.getElementById("batch-progress-panel");
  batchPanel.classList.remove("hidden");
  document.getElementById("batch-summary-table").innerHTML = "";
  setBatchProgress("starting", 0, "", 0, state.batchFiles.length);

  const res = await fetch("/api/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths: state.batchFiles.map((f) => f.path), preset }),
  });
  if (!res.ok) {
    alert("Failed to start batch: " + (await res.text()));
    document.getElementById("run-button").disabled = false;
    return;
  }
  const { batch_id } = await res.json();
  state.batchId = batch_id;

  const es = new EventSource(`/api/batch/${batch_id}/events`);
  es.onmessage = (e) => {
    const item = JSON.parse(e.data);
    if (item.type === "progress") {
      setBatchProgress(item.stage, item.progress, item.filename, item.index, item.total);
    } else if (item.type === "done") {
      es.close();
      document.getElementById("run-button").disabled = false;
      renderBatchSummary(item.summary);
      loadHistory();
    }
  };
}

function setBatchProgress(stage, frac, filename, index, total) {
  const fileLabel = filename ? ` (${filename})` : "";
  document.getElementById("batch-stage-label").textContent =
    `File ${index + 1}/${total}${fileLabel}: ${stage} - ${Math.round(frac * 100)}%`;
  document.getElementById("batch-progress-fill").style.width = `${Math.round(frac * 100)}%`;
}

function renderBatchSummary(summary) {
  const container = document.getElementById("batch-summary-table");
  container.innerHTML = summary
    .map((s) => {
      const cls = s.status === "done" ? "status-done" : "status-error";
      const detail =
        s.status === "done"
          ? `<a href="#" data-job="${s.job_id}" class="view-job-link">view result</a>`
          : s.error;
      return `<div class="batch-summary-row ${cls}"><span>${s.filename}</span><span>${detail}</span></div>`;
    })
    .join("");
  container.querySelectorAll(".view-job-link").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      loadResults(e.currentTarget.dataset.job);
    });
  });
}

// ---- Custom presets ----

function setupSavePresetButton() {
  document.getElementById("save-preset-button").addEventListener("click", async () => {
    const currentKey = document.getElementById("preset-select").value;
    const current = state.presets[currentKey];
    if (!current) return;
    const name = prompt("New preset name:", current.name + " Copy");
    if (!name) return;

    const res = await fetch("/api/presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        description: current.description,
        stages: current.stages,
        finishing: current.finishing,
      }),
    });
    if (!res.ok) {
      alert("Failed to save preset: " + (await res.text()));
      return;
    }
    const { key } = await res.json();
    await loadPresets();
    document.getElementById("preset-select").value = key;
    updatePresetDescription();
  });
}

// ---- Job history ----

async function loadHistory() {
  const container = document.getElementById("history-list");
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const jobs = await res.json();
    if (!jobs.length) {
      container.innerHTML = '<p class="detail">No jobs yet.</p>';
      return;
    }
    container.innerHTML = jobs
      .map((j) => {
        const when = j.created_utc ? new Date(j.created_utc).toLocaleString() : "";
        const flag = j.integrity_flagged ? " ⚠" : "";
        return `
          <div class="history-row" data-job="${j.job_id}">
            <span>${j.input_filename}${flag}</span>
            <span class="detail">${j.preset || ""} · ${when}</span>
          </div>
        `;
      })
      .join("");
    container.querySelectorAll(".history-row").forEach((row) => {
      row.addEventListener("click", () => loadResults(row.dataset.job));
    });
  } catch (e) {
    container.innerHTML = '<p class="detail">Failed to load job history.</p>';
    console.error("loadHistory failed:", e);
  }
}

// ---- Results + gapless loudness-matched A/B player ----

const player = {
  ctx: null,
  origGain: null,
  finalGain: null,
  active: "final",
  origMatch: 1,
  finalMatch: 1,
};

const LOUDNESS_REFERENCE = -23; // arbitrary common reference used only for fair A/B comparison

async function loadResults(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  const data = await res.json();
  if (!data.job) return;

  const origLufs = data.job.analysis.lufs;
  const finalLufs = data.job.final_lufs;

  setupPlayer(jobId, origLufs, finalLufs);

  document.getElementById("loudness-info").textContent =
    `Original: ${origLufs.toFixed(1)} LUFS -> Final: ${finalLufs.toFixed(1)} LUFS ` +
    `(A/B playback level-matched for fair comparison)`;

  const downloadLink = document.getElementById("download-link");
  downloadLink.href = `/api/jobs/${jobId}/download?which=final`;

  const videoLink = document.getElementById("download-video-link");
  if (data.job.output_video) {
    videoLink.href = `/api/jobs/${jobId}/download?which=video`;
    videoLink.classList.remove("hidden");
  } else {
    videoLink.classList.add("hidden");
  }

  renderMetricsTable(data.job.metrics || []);
  renderIntegrityWarning(data.job.integrity);

  document.getElementById("results-panel").classList.remove("hidden");
}

function renderIntegrityWarning(integrity) {
  const container = document.getElementById("integrity-warning");
  if (!integrity || !integrity.ran || !integrity.flagged) {
    container.innerHTML = "";
    return;
  }

  const segmentRows = integrity.differing_segments
    .map(
      (s) =>
        `<li>${s.start.toFixed(1)}s-${s.end.toFixed(1)}s: "${s.before_text}" -> "${s.after_text}"</li>`
    )
    .join("");

  container.innerHTML = `
    <div class="integrity-warning">
      <strong>Content integrity check flagged this job</strong>
      <p class="detail">Word error rate ${(integrity.wer * 100).toFixed(1)}% between the audio before and after
      generative restoration. This can mean the generative step changed or invented words - listen to the
      flagged segments before trusting this output.</p>
      <ul>${segmentRows}</ul>
    </div>
  `;
}

function renderMetricsTable(metrics) {
  const container = document.getElementById("metrics-table");
  if (!metrics.length) {
    container.innerHTML = '<p class="detail">DNSMOS scoring is not available (dnsmos engine not installed).</p>';
    return;
  }

  let prevOvrl = null;
  const rows = metrics
    .map((m) => {
      const flagged = prevOvrl !== null && m.ovrl !== null && prevOvrl - m.ovrl > 0.2;
      const rowClass = flagged ? ' class="metric-row-flagged"' : "";
      prevOvrl = m.ovrl;
      const fmt = (v) => (v === null || v === undefined ? "-" : v.toFixed(2));
      return `<tr${rowClass}><td>${m.stage}</td><td>${fmt(m.sig)}</td><td>${fmt(m.bak)}</td><td>${fmt(m.ovrl)}</td></tr>`;
    })
    .join("");

  container.innerHTML = `
    <table class="metrics-table">
      <thead><tr><th>Stage</th><th>SIG</th><th>BAK</th><th>OVRL</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="detail">DNSMOS P.835 (higher is better, 1-5 scale). Rows in orange dropped OVRL by more than 0.2 from the previous stage.</p>
  `;
}

function setupPlayer(jobId, origLufs, finalLufs) {
  const origAudio = document.getElementById("orig-audio");
  const finalAudio = document.getElementById("final-audio");
  origAudio.src = `/api/jobs/${jobId}/audio/original`;
  finalAudio.src = `/api/jobs/${jobId}/audio/final`;
  origAudio.load();
  finalAudio.load();

  if (!player.ctx) {
    player.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const origSource = player.ctx.createMediaElementSource(origAudio);
    const finalSource = player.ctx.createMediaElementSource(finalAudio);
    player.origGain = player.ctx.createGain();
    player.finalGain = player.ctx.createGain();
    origSource.connect(player.origGain).connect(player.ctx.destination);
    finalSource.connect(player.finalGain).connect(player.ctx.destination);
  }

  player.origMatch = Math.pow(10, (LOUDNESS_REFERENCE - origLufs) / 20);
  player.finalMatch = Math.pow(10, (LOUDNESS_REFERENCE - finalLufs) / 20);

  setActiveTrack("final");

  finalAudio.addEventListener("timeupdate", () => {
    if (finalAudio.duration) {
      document.getElementById("seek-bar").value = String((finalAudio.currentTime / finalAudio.duration) * 1000);
    }
  });
}

function setActiveTrack(which) {
  player.active = which;
  const origAudio = document.getElementById("orig-audio");
  const finalAudio = document.getElementById("final-audio");
  player.origGain.gain.value = which === "original" ? player.origMatch : 0;
  player.finalGain.gain.value = which === "final" ? player.finalMatch : 0;

  document.getElementById("ab-original").classList.toggle("active", which === "original");
  document.getElementById("ab-final").classList.toggle("active", which === "final");

  // Keep both transports in sync even while muted, so switching is instant and gapless.
  if (Math.abs(origAudio.currentTime - finalAudio.currentTime) > 0.05) {
    origAudio.currentTime = finalAudio.currentTime;
  }
}

function togglePlay() {
  const origAudio = document.getElementById("orig-audio");
  const finalAudio = document.getElementById("final-audio");
  if (player.ctx.state === "suspended") player.ctx.resume();

  if (finalAudio.paused) {
    origAudio.currentTime = finalAudio.currentTime;
    origAudio.play();
    finalAudio.play();
    document.getElementById("play-button").textContent = "Pause";
  } else {
    origAudio.pause();
    finalAudio.pause();
    document.getElementById("play-button").textContent = "Play";
  }
}

function setupPlayerControls() {
  document.getElementById("play-button").addEventListener("click", togglePlay);
  document.getElementById("ab-original").addEventListener("click", () => setActiveTrack("original"));
  document.getElementById("ab-final").addEventListener("click", () => setActiveTrack("final"));

  document.getElementById("seek-bar").addEventListener("input", (e) => {
    const finalAudio = document.getElementById("final-audio");
    const origAudio = document.getElementById("orig-audio");
    if (!finalAudio.duration) return;
    const t = (Number(e.target.value) / 1000) * finalAudio.duration;
    finalAudio.currentTime = t;
    origAudio.currentTime = t;
  });

  document.addEventListener("keydown", (e) => {
    if (document.getElementById("results-panel").classList.contains("hidden")) return;
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "a" || e.key === "A") setActiveTrack("original");
    else if (e.key === "b" || e.key === "B") setActiveTrack("final");
    else if (e.key === " ") {
      e.preventDefault();
      togglePlay();
    }
  });
}

// ---- Init ----

loadHealth();
loadPresets();
loadHistory();
setupDropzone();
setupRunButton();
setupSavePresetButton();
setupPlayerControls();
