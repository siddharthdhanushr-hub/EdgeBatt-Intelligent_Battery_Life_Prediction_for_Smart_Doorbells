/**
 * EdgeBatt Web Controller — EDGI AI Team 01
 * Smart Doorbell Battery Intelligence Dashboard
 */

document.addEventListener('DOMContentLoaded', () => {
  // Base API endpoint resolver (supports direct server URL, Codespaces, and file:/// previews)
  const isDirectOrCodespaces = window.location.port === '8000' || window.location.hostname.includes('github.dev');
  const API_BASE = isDirectOrCodespaces ? '' : 'http://127.0.0.1:8000';

  // Preset Telemetry Configurations
  const PRESETS = {
    fresh: {
      battery_v: 4.15,
      temp_c: 22.0,
      rings_per_day: 3,
      wifi_rssi_dbm: -55,
      age_months: 1
    },
    residential: {
      battery_v: 3.86,
      temp_c: 28.0,
      rings_per_day: 5,
      wifi_rssi_dbm: -66,
      age_months: 22
    },
    winter: {
      battery_v: 3.60,
      temp_c: -5.0,
      rings_per_day: 15,
      wifi_rssi_dbm: -75,
      age_months: 18
    },
    depleted: {
      battery_v: 3.35,
      temp_c: 35.0,
      rings_per_day: 25,
      wifi_rssi_dbm: -85,
      age_months: 34
    },
    glitch: {
      battery_v: 3.86,
      temp_c: -127.0,
      rings_per_day: 5,
      wifi_rssi_dbm: 0,
      age_months: 22
    }
  };

  // Pre-configured Fleet Telemetry for Batch Testing
  const FLEET_DEVICES = [
    { id: "Front Porch (DB-01)", battery_v: 3.86, temp_c: 28.0, rings_per_day: 5, wifi_rssi_dbm: -66, age_months: 22 },
    { id: "Courtyard Gate (DB-02)", battery_v: 4.10, temp_c: 22.0, rings_per_day: 2, wifi_rssi_dbm: -50, age_months: 5 },
    { id: "Back Alley Service (DB-03)", battery_v: 3.45, temp_c: 35.0, rings_per_day: 18, wifi_rssi_dbm: -82, age_months: 29 }
  ];

  // DOM Elements
  const fields = ['battery_v', 'temp_c', 'rings_per_day', 'wifi_rssi_dbm', 'age_months'];
  const sensorForm = document.getElementById('sensorForm');
  const btnPredict = document.getElementById('btnPredict');
  const predictSpinner = document.getElementById('predictSpinner');
  const btnReset = document.getElementById('btnReset');
  const btnRunFleet = document.getElementById('btnRunFleet');
  const errorBanner = document.getElementById('errorBanner');
  const errorTitle = document.getElementById('errorTitle');
  const errorMsg = document.getElementById('errorMsg');

  // Prediction Display Elements
  const predDays = document.getElementById('predDays');
  const healthBadge = document.getElementById('healthBadge');
  const batteryLevelBar = document.getElementById('batteryLevelBar');
  const predDepletionDate = document.getElementById('predDepletionDate');
  const predAction = document.getElementById('predAction');

  // Timing Elements
  const totalLatencyBadge = document.getElementById('totalLatencyBadge');
  const valParse = document.getElementById('valParse');
  const valNorm = document.getElementById('valNorm');
  const valInfer = document.getElementById('valInfer');
  const valBuild = document.getElementById('valBuild');
  const barParse = document.getElementById('barParse');
  const barNorm = document.getElementById('barNorm');
  const barInfer = document.getElementById('barInfer');
  const barBuild = document.getElementById('barBuild');

  // Header Status Elements
  const beaconDot = document.getElementById('beaconDot');
  const apiStatusText = document.getElementById('apiStatusText');
  const statParams = document.getElementById('statParams');
  const statValMae = document.getElementById('statValMae');

  // Hardware Elements
  const sramStatusBadge = document.getElementById('sramStatusBadge');
  const sramUsageText = document.getElementById('sramUsageText');
  const sramProgressFill = document.getElementById('sramProgressFill');
  const layersTableBody = document.getElementById('layersTableBody');
  const fleetTableBody = document.getElementById('fleetTableBody');
  const fleetTimingText = document.getElementById('fleetTimingText');
  const wifiBars = document.getElementById('wifiBars');

  // 1. Two-way Bind Range Sliders with Numeric Inputs
  fields.forEach(f => {
    const numInput = document.getElementById(f);
    const slider = document.getElementById(`${f}_slider`);

    if (numInput && slider) {
      numInput.addEventListener('input', () => {
        slider.value = numInput.value;
        if (f === 'wifi_rssi_dbm') updateWifiSignalBars(parseFloat(numInput.value));
      });

      slider.addEventListener('input', () => {
        numInput.value = slider.value;
        if (f === 'wifi_rssi_dbm') updateWifiSignalBars(parseFloat(slider.value));
      });
    }
  });

  // Update Wi-Fi signal icon bars
  function updateWifiSignalBars(rssi) {
    wifiBars.className = 'wifi-bars';
    if (isNaN(rssi) || rssi < -85) {
      wifiBars.classList.add('lvl-1');
    } else if (rssi < -70) {
      wifiBars.classList.add('lvl-2');
    } else if (rssi < -55) {
      wifiBars.classList.add('lvl-3');
    } else {
      wifiBars.classList.add('lvl-4');
    }
  }

  // 2. Preset Loader
  function applyPreset(presetKey) {
    const preset = PRESETS[presetKey];
    if (!preset) return;

    fields.forEach(f => {
      const numInput = document.getElementById(f);
      const slider = document.getElementById(`${f}_slider`);
      if (numInput && preset[f] !== undefined) {
        numInput.value = preset[f];
        if (slider) slider.value = preset[f];
      }
    });

    updateWifiSignalBars(preset.wifi_rssi_dbm);

    // Update active preset chip
    document.querySelectorAll('.preset-chip').forEach(chip => {
      chip.classList.toggle('active', chip.dataset.preset === presetKey);
    });

    // Auto-predict on preset selection
    executePredict();
  }

  // Preset button listeners
  document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.preventDefault();
      applyPreset(chip.dataset.preset);
    });
  });

  btnReset.addEventListener('click', () => {
    applyPreset('residential');
  });

  // 3. API Health Heartbeat Polling
  async function checkHealth() {
    const t0 = performance.now();
    try {
      const res = await fetch(`${API_BASE}/health`);
      const ping = Math.round(performance.now() - t0);
      if (res.ok) {
        const data = await res.json();
        beaconDot.className = 'beacon-dot beacon-connected';
        apiStatusText.textContent = `API Connected (${ping}ms) • Team ${String(data.team).padStart(2, '0')}`;
        statParams.textContent = Number(data.params).toLocaleString();
        statValMae.textContent = `${data.val_mae} d`;
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      beaconDot.className = 'beacon-dot beacon-disconnected';
      apiStatusText.textContent = 'Backend Offline';
      statParams.textContent = '--';
      statValMae.textContent = '--';
    }
  }

  // 4. Fetch Dynamic Model Architecture Specs (/model_info)
  async function fetchModelInfo() {
    try {
      const res = await fetch(`${API_BASE}/model_info`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // SRAM Progress Meter
      const sramLimit = 300 * 1024; // 307,200 bytes
      const percent = ((data.total_bytes_f32 / sramLimit) * 100).toFixed(2);
      sramProgressFill.style.width = `${percent}%`;
      sramUsageText.textContent = `${data.total_bytes_f32.toLocaleString()} / ${sramLimit.toLocaleString()} Bytes (${percent}%)`;
      sramStatusBadge.textContent = data.fits_300kb_sram_f32 ? 'Fits 300 KB SRAM' : 'Exceeds 300 KB SRAM';

      // Layers Table
      if (data.layers && data.layers.length > 0) {
        layersTableBody.innerHTML = data.layers.map(l => `
          <tr>
            <td style="font-weight:600; color:var(--accent-cyan);">${l.name}</td>
            <td>${l.in} features</td>
            <td>${l.out} units</td>
            <td>${l.params.toLocaleString()}</td>
            <td>${l.bytes_f32.toLocaleString()} B</td>
            <td>${l.macs.toLocaleString()}</td>
          </tr>
        `).join('');

        // Append total summary row
        layersTableBody.innerHTML += `
          <tr style="background:rgba(6, 182, 212, 0.05); font-weight:700;">
            <td style="color:var(--text-main);">TOTAL NETWORK</td>
            <td>5 features</td>
            <td>1 output</td>
            <td style="color:var(--accent-cyan);">${data.total_params.toLocaleString()}</td>
            <td style="color:var(--accent-emerald);">${data.total_bytes_f32.toLocaleString()} B</td>
            <td style="color:var(--accent-cyan);">${data.total_macs.toLocaleString()}</td>
          </tr>
        `;
      }
    } catch (err) {
      console.warn("Could not fetch /model_info:", err);
    }
  }

  // 5. Predict Single Reading (POST /predict)
  async function executePredict() {
    // Gather form readings
    const reading = {};
    for (const f of fields) {
      const val = parseFloat(document.getElementById(f).value);
      reading[f] = val;
    }

    // UI Loading state
    btnPredict.disabled = true;
    predictSpinner.classList.remove('hidden');
    errorBanner.classList.add('hidden');

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reading)
      });

      if (res.status === 422) {
        // Physical bounds violation / Hardware glitch rejected
        const errJson = await res.json();
        renderValidationError(errJson);
        return;
      }

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      renderPredictionResult(data.days_left, data.timings_ms, reading);

    } catch (err) {
      errorBanner.classList.remove('hidden');
      errorTitle.textContent = "Prediction Request Failed";
      errorMsg.textContent = err.message || "Failed to communicate with FastAPI backend.";
    } finally {
      btnPredict.disabled = false;
      predictSpinner.classList.add('hidden');
    }
  }

  // Render HTTP 422 Error
  function renderValidationError(errJson) {
    errorBanner.classList.remove('hidden');
    errorTitle.textContent = "Sensor Telemetry Rejected (HTTP 422)";

    let details = "Sensor readings are out of realistic physical range or hardware fault detected.";
    if (errJson.detail && Array.isArray(errJson.detail)) {
      const issues = errJson.detail.map(d => {
        const fieldName = d.loc ? d.loc.slice(-1)[0] : 'field';
        return `<strong>${fieldName}</strong>: ${d.msg}`;
      }).join('<br>');
      details = issues;
    }
    errorMsg.innerHTML = details;

    // Reset visual indicators to error warning state
    predDays.textContent = "ERR 422";
    healthBadge.className = 'badge-status badge-critical';
    healthBadge.textContent = 'Faulty Input';
    batteryLevelBar.className = 'battery-level-bar critical';
    batteryLevelBar.style.width = '10%';
    predDepletionDate.textContent = 'N/A (Fault)';
    predAction.textContent = 'Inspect sensor telemetry wiring';
  }

  // Render Successful Prediction
  function renderPredictionResult(days, timings, reading) {
    // Smooth number display
    predDays.textContent = days.toFixed(2);

    // Battery cell fill calculation (assuming 180 days is nominal full lifespan)
    const fillPercent = Math.min(Math.max((days / 180) * 100, 6), 100);
    batteryLevelBar.style.width = `${fillPercent}%`;

    // Health Badge and Styling
    if (days >= 60) {
      healthBadge.className = 'badge-status badge-healthy';
      healthBadge.textContent = 'Healthy';
      batteryLevelBar.className = 'battery-level-bar';
      predAction.textContent = 'Normal monitoring; no action required';
    } else if (days >= 20) {
      healthBadge.className = 'badge-status badge-moderate';
      healthBadge.textContent = 'Moderate';
      batteryLevelBar.className = 'battery-level-bar moderate';
      predAction.textContent = 'Schedule battery recharge within ~3 weeks';
    } else {
      healthBadge.className = 'badge-status badge-critical';
      healthBadge.textContent = 'Critical Low';
      batteryLevelBar.className = 'battery-level-bar critical';
      predAction.textContent = 'Immediate battery recharge / replacement advised';
    }

    // Depletion Date Calculation
    const targetDate = new Date();
    targetDate.setDate(targetDate.getDate() + Math.round(days));
    predDepletionDate.textContent = targetDate.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });

    // Render Micro-Latency Stages
    if (timings) {
      const parse = timings.parse || 0;
      const norm = timings.normalise || 0;
      const infer = timings.inference || 0;
      const build = timings.build || 0;
      const total = timings.server_total || (parse + norm + infer + build);

      totalLatencyBadge.textContent = `${total.toFixed(3)} ms`;
      valParse.textContent = `${parse.toFixed(3)} ms`;
      valNorm.textContent = `${norm.toFixed(3)} ms`;
      valInfer.textContent = `${infer.toFixed(3)} ms`;
      valBuild.textContent = `${build.toFixed(3)} ms`;

      // Relative bar percentages
      const sum = Math.max(parse + norm + infer + build, 0.001);
      barParse.style.width = `${Math.max((parse / sum) * 100, 4)}%`;
      barNorm.style.width = `${Math.max((norm / sum) * 100, 4)}%`;
      barInfer.style.width = `${Math.max((infer / sum) * 100, 10)}%`;
      barBuild.style.width = `${Math.max((build / sum) * 100, 4)}%`;
    }
  }

  // 6. IoT Fleet Batch Prediction (POST /predict_batch)
  async function executeFleetBatch() {
    btnRunFleet.disabled = true;
    fleetTimingText.textContent = "Processing batch...";

    try {
      const payload = {
        rows: FLEET_DEVICES.map(d => ({
          battery_v: d.battery_v,
          temp_c: d.temp_c,
          rings_per_day: d.rings_per_day,
          wifi_rssi_dbm: d.wifi_rssi_dbm,
          age_months: d.age_months
        }))
      };

      const res = await fetch(`${API_BASE}/predict_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      fleetTableBody.innerHTML = FLEET_DEVICES.map((d, i) => {
        const days = data.days_left[i];
        let statusBadge = '<span class="badge-status badge-healthy">Healthy</span>';
        if (days < 20) {
          statusBadge = '<span class="badge-status badge-critical">Critical</span>';
        } else if (days < 60) {
          statusBadge = '<span class="badge-status badge-moderate">Moderate</span>';
        }

        return `
          <tr>
            <td style="font-weight:600; color:var(--text-main); font-family:var(--font-sans);">${d.id}</td>
            <td>${d.battery_v.toFixed(2)} V</td>
            <td>${d.temp_c.toFixed(1)} °C</td>
            <td>${d.rings_per_day}</td>
            <td>${d.wifi_rssi_dbm} dBm</td>
            <td>${d.age_months} mo</td>
            <td style="color:var(--accent-cyan); font-weight:700;">${days.toFixed(2)} days</td>
            <td>${statusBadge}</td>
          </tr>
        `;
      }).join('');

      const t = data.timings_ms;
      fleetTimingText.textContent = `${t.server_total.toFixed(3)} ms total (${(t.server_total / FLEET_DEVICES.length).toFixed(3)} ms / device)`;

    } catch (err) {
      fleetTimingText.textContent = `Error: ${err.message}`;
    } finally {
      btnRunFleet.disabled = false;
    }
  }

  // Event Listeners
  sensorForm.addEventListener('submit', (e) => {
    e.preventDefault();
    executePredict();
  });

  btnRunFleet.addEventListener('click', () => {
    executeFleetBatch();
  });

  // Initialization
  applyPreset('residential');
  checkHealth();
  fetchModelInfo();
  executeFleetBatch();

  // Periodic health check every 8 seconds
  setInterval(checkHealth, 8000);
});

