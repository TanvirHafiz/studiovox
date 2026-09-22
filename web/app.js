async function loadHealth() {
  const list = document.getElementById("status-list");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    list.innerHTML = "";

    list.appendChild(row("ffmpeg", data.ffmpeg.found, data.ffmpeg.version || data.ffmpeg.error));

    const gpuDetail = data.gpu.found
      ? `${data.gpu.name} (${(data.gpu.vram_total_mb / 1024).toFixed(1)} GB VRAM, driver ${data.gpu.driver_version})`
      : data.gpu.error;
    list.appendChild(row("GPU", data.gpu.found, gpuDetail, data.gpu.found ? "ok" : "warn"));

    for (const disk of data.disk) {
      const state = disk.low_space_warning ? "warn" : "ok";
      list.appendChild(row(`disk (${disk.path})`, true, `${disk.free_gb} GB free of ${disk.total_gb} GB`, state));
    }
  } catch (e) {
    list.innerHTML = `<div class="status-row"><span class="dot err"></span>Could not reach server: ${e}</div>`;
  }
}

function row(label, ok, detail, forceState) {
  const div = document.createElement("div");
  div.className = "status-row";
  const state = forceState || (ok ? "ok" : "err");
  div.innerHTML = `<span><span class="dot ${state}"></span>${label}</span><span class="detail">${detail || ""}</span>`;
  return div;
}

loadHealth();
