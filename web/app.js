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
};

// ---- Presets ----

async function loadPresets() {
  const res = await fetch("/api/presets");
  state.presets = await res.json();
  const select = document.getElementById("preset-select");
  select.innerHTML = "";
  for (const [key, preset] of Object.entries(state.presets)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = preset.name;
    select.appendChild(opt);
  }
  updatePresetDescription();
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

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) handleFile(input.files[0]);
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
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
}

async function handleFile(file) {
  resetForNewFile();

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

function resetForNewFile() {
  document.getElementById("results-panel").classList.add("hidden");
  document.getElementById("progress-panel").classList.add("hidden");
  state.jobId = null;
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
  document.getElementById("run-button").addEventListener("click", runJob);
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

  document.getElementById("results-panel").classList.remove("hidden");
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
setupDropzone();
setupRunButton();
setupPlayerControls();
