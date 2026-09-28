/**
 * VisionAttend AI - Smart Face Recognition Attendance System
 * Client-Side Neural Biometric Inference Engine & Management Dashboard
 */

// ------------------ STATE MANAGEMENT ------------------
const state = {
  modelsLoaded: false,
  isCameraRunning: false,
  stream: null,
  facingMode: 'user',
  isMirrored: true,
  faceMatcher: null,
  registeredProfiles: [],
  attendanceRecords: [],
  lastAttendanceDebounce: {}, // name -> timestamp
  currentDetector: 'tiny', // 'tiny' or 'ssd'
  recognitionInterval: null,
  enrollStream: null,
  currentEnrollDescriptor: null,
  currentEnrollImageBase64: null,
  audioCtx: null
};

// ------------------ AUDIO CHIME SYNTHESIZER ------------------
function playSuccessChime() {
  try {
    if (!state.audioCtx) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }
    const now = state.audioCtx.currentTime;
    
    // Primary Tone
    const osc1 = state.audioCtx.createOscillator();
    const gain1 = state.audioCtx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(587.33, now); // D5
    osc1.frequency.exponentialRampToValueAtTime(880, now + 0.15); // A5
    gain1.gain.setValueAtTime(0.2, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc1.connect(gain1);
    gain1.connect(state.audioCtx.destination);
    osc1.start(now);
    osc1.stop(now + 0.35);

    // Harmonic Bell Tone
    const osc2 = state.audioCtx.createOscillator();
    const gain2 = state.audioCtx.createGain();
    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(1174.66, now + 0.08); // D6
    gain2.gain.setValueAtTime(0.15, now + 0.08);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
    osc2.connect(gain2);
    gain2.connect(state.audioCtx.destination);
    osc2.start(now + 0.08);
    osc2.stop(now + 0.45);
  } catch (err) {
    console.warn("Audio chime disabled or blocked by browser:", err);
  }
}

// ------------------ CLOCK & DATE ------------------
function initLiveClock() {
  const clockEl = document.getElementById('liveClock');
  const dateEl = document.getElementById('liveDate');

  function update() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    dateEl.textContent = now.toLocaleDateString('en-US', {
      weekday: 'short', month: 'short', day: 'numeric', year: 'numeric'
    });
  }
  update();
  setInterval(update, 1000);
}

// ------------------ INITIALIZE REGISTERED PROFILES ------------------
function loadProfiles() {
  const preloaded = (window.PRELOADED_REGISTERED_FACES || []).map(p => ({
    name: p.name,
    image: p.image,
    descriptor: new Float32Array(p.descriptor),
    role: 'Student / Personnel',
    id: `REG-2026-0${Math.floor(Math.random() * 900 + 100)}`
  }));

  let custom = [];
  try {
    const raw = localStorage.getItem('CUSTOM_FACES');
    if (raw) {
      const parsed = JSON.parse(raw);
      custom = parsed.map(p => ({
        name: p.name,
        image: p.image,
        descriptor: new Float32Array(p.descriptor),
        role: p.role || 'Member',
        id: p.id || 'ID-NEW'
      }));
    }
  } catch (e) {
    console.error("Failed to parse custom faces from storage", e);
  }

  // Merge avoiding duplicates by name
  const combined = [...preloaded];
  custom.forEach(c => {
    if (!combined.some(existing => existing.name.toLowerCase() === c.name.toLowerCase())) {
      combined.push(c);
    }
  });

  state.registeredProfiles = combined;
  updateFaceMatcher();
  renderPersonnelDirectory();
  renderDemoButtons();
  updateStats();
}

function updateFaceMatcher() {
  if (!window.faceapi || state.registeredProfiles.length === 0) return;

  try {
    const labeledDescriptors = state.registeredProfiles.map(p => {
      return new faceapi.LabeledFaceDescriptors(p.name, [p.descriptor]);
    });
    // Tolerance 0.55 for reliable recognition
    state.faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.55);
    console.log(`[FaceMatcher] Configured with ${labeledDescriptors.length} registered profiles`);
  } catch (err) {
    console.error("Error setting up FaceMatcher:", err);
  }
}

// ------------------ INITIALIZE ATTENDANCE LOGS ------------------
function loadAttendanceLogs() {
  let logs = [];
  try {
    const stored = localStorage.getItem('ATTENDANCE_LOGS');
    if (stored) {
      logs = JSON.parse(stored);
    } else {
      // Default initial mock history matching attendance.csv
      const todayStr = new Date().toISOString().split('T')[0];
      logs = [
        { name: "Ipsita", date: todayStr, time: "11:16:07", status: "Present", confidence: 96 },
        { name: "Kanishka", date: todayStr, time: "11:16:25", status: "Present", confidence: 94 },
        { name: "Tanvi", date: todayStr, time: "15:11:27", status: "Present", confidence: 97 },
        { name: "Senna", date: todayStr, time: "14:18:46", status: "Present", confidence: 93 }
      ];
      localStorage.setItem('ATTENDANCE_LOGS', JSON.stringify(logs));
    }
  } catch (err) {
    console.error("Error loading attendance logs:", err);
  }

  state.attendanceRecords = logs;
  renderAttendanceTable();
  updateStats();
}

function markAttendance(name, confidence = 95) {
  const now = new Date();
  const dateStr = now.toISOString().split('T')[0];
  const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
  const normalizedName = name.trim();

  // Debounce check: prevent multiple triggers within 10 seconds
  const lastTime = state.lastAttendanceDebounce[normalizedName.toLowerCase()] || 0;
  if (Date.now() - lastTime < 10000) {
    return false;
  }
  state.lastAttendanceDebounce[normalizedName.toLowerCase()] = Date.now();

  // Check if already logged today
  const alreadyLoggedToday = state.attendanceRecords.some(
    r => r.name.toLowerCase() === normalizedName.toLowerCase() && r.date === dateStr
  );

  // Play audio chime
  playSuccessChime();

  // Show live toast banner
  showDetectionAlert(normalizedName, confidence, alreadyLoggedToday);

  if (alreadyLoggedToday) {
    console.log(`[Attendance] ${normalizedName} already verified for today (${dateStr})`);
    return false;
  }

  const record = {
    name: normalizedName,
    date: dateStr,
    time: timeStr,
    status: "Present",
    confidence: Math.round(confidence)
  };

  // Add to top of records
  state.attendanceRecords.unshift(record);
  localStorage.setItem('ATTENDANCE_LOGS', JSON.stringify(state.attendanceRecords));

  // Sync to Flask backend if running locally
  fetch('/api/attendance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(record)
  }).catch(() => {/* Offline / Pure client fallback */});

  renderAttendanceTable();
  updateStats();
  return true;
}

function showDetectionAlert(name, confidence, alreadyMarked = false) {
  const alertEl = document.getElementById('detectionAlert');
  const nameEl = document.getElementById('alertName');
  const subEl = document.getElementById('alertSub');
  const badgeEl = document.getElementById('alertBadge');
  const avatarEl = document.getElementById('alertAvatar');

  nameEl.textContent = name;
  subEl.textContent = alreadyMarked
    ? `Verified Again • ${Math.round(confidence)}% Match (Already Logged)`
    : `Biometric Match • ${Math.round(confidence)}% Match • Verified Present!`;

  badgeEl.textContent = alreadyMarked ? 'Verified' : 'Logged Present';

  // Find profile image
  const profile = state.registeredProfiles.find(p => p.name.toLowerCase() === name.toLowerCase());
  if (profile && profile.image) {
    avatarEl.innerHTML = `<img src="${profile.image}" alt="${name}">`;
  } else {
    avatarEl.textContent = '👤';
  }

  alertEl.classList.add('visible');
  clearTimeout(alertEl._timeout);
  alertEl._timeout = setTimeout(() => {
    alertEl.classList.remove('visible');
  }, 4000);
}

// ------------------ RENDER UI COMPONENTS ------------------
function updateStats() {
  const registeredCountEl = document.getElementById('statRegisteredCount');
  const presentCountEl = document.getElementById('statPresentCount');
  const presentPctEl = document.getElementById('statPresentPct');
  const lastTimeEl = document.getElementById('statLastTime');
  const lastNameEl = document.getElementById('statLastName');

  const todayStr = new Date().toISOString().split('T')[0];
  const totalRegistered = state.registeredProfiles.length;
  registeredCountEl.textContent = totalRegistered;

  // Calculate distinct personnel present today
  const presentTodaySet = new Set(
    state.attendanceRecords
      .filter(r => r.date === todayStr)
      .map(r => r.name.toLowerCase())
  );

  const presentCount = presentTodaySet.size;
  presentCountEl.textContent = presentCount;
  const pct = totalRegistered > 0 ? Math.round((presentCount / totalRegistered) * 100) : 0;
  presentPctEl.textContent = `${pct}% of registered team`;

  if (state.attendanceRecords.length > 0) {
    const last = state.attendanceRecords[0];
    lastTimeEl.textContent = last.time;
    lastNameEl.textContent = `${last.name} (${last.date === todayStr ? 'Today' : last.date})`;
  } else {
    lastTimeEl.textContent = '--:--';
    lastNameEl.textContent = 'No scan yet';
  }
}

function renderAttendanceTable() {
  const tbody = document.getElementById('attendanceTbody');
  const emptyState = document.getElementById('emptyTableState');
  const searchVal = (document.getElementById('searchFilter').value || '').trim().toLowerCase();
  const dateFilter = document.getElementById('dateFilter').value;
  const todayStr = new Date().toISOString().split('T')[0];

  let filtered = state.attendanceRecords.filter(r => {
    const matchName = r.name.toLowerCase().includes(searchVal);
    const matchDate = dateFilter === 'today' ? (r.date === todayStr) : true;
    return matchName && matchDate;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = '';
    emptyState.style.display = 'flex';
    return;
  }

  emptyState.style.display = 'none';
  tbody.innerHTML = filtered.map(r => {
    const profile = state.registeredProfiles.find(p => p.name.toLowerCase() === r.name.toLowerCase());
    const avatarHtml = (profile && profile.image)
      ? `<img src="${profile.image}" class="person-avatar" alt="${r.name}">`
      : `<div class="person-avatar">${r.name.charAt(0)}</div>`;

    return `
      <tr>
        <td>
          <div class="person-cell">
            ${avatarHtml}
            <span>${r.name}</span>
          </div>
        </td>
        <td>${r.date}</td>
        <td><strong style="color:var(--text-primary); font-variant-numeric:tabular-nums;">${r.time}</strong></td>
        <td><span class="status-badge present">✓ ${r.status || 'Present'}</span></td>
        <td><span style="color:var(--accent-cyan); font-weight:600;">${r.confidence ? r.confidence + '%' : '95%'}</span></td>
      </tr>
    `;
  }).join('');
}

function renderPersonnelDirectory() {
  const container = document.getElementById('personnelGrid');
  if (!container) return;

  container.innerHTML = state.registeredProfiles.map(p => {
    return `
      <div class="person-card">
        <img src="${p.image}" class="person-card-avatar" alt="${p.name}" onerror="this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(p.name)}&background=00e6b8&color=050811'">
        <div class="person-card-info">
          <div class="person-card-name">${p.name}</div>
          <div class="person-card-role">${p.role || 'Active Member'} &bull; ${p.id || 'REG'}</div>
          <div class="person-card-status">Active Biometrics</div>
        </div>
        <button class="person-card-btn" onclick="triggerTestPerson('${p.name}')">Simulate</button>
      </div>
    `;
  }).join('');
}

function renderDemoButtons() {
  const container = document.getElementById('demoButtons');
  if (!container) return;

  container.innerHTML = state.registeredProfiles.map(p => {
    return `
      <button class="demo-btn" onclick="triggerTestPerson('${p.name}')">
        <img src="${p.image}" alt="${p.name}" onerror="this.style.display='none'">
        <span>${p.name}</span>
      </button>
    `;
  }).join('');
}

// ------------------ CAMERA ENGINE ------------------
const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('overlayCanvas');
const placeholder = document.getElementById('cameraPlaceholder');
const laser = document.getElementById('scanLaser');
const cameraBadge = document.getElementById('cameraStatusBadge');

async function startCamera() {
  try {
    cameraBadge.textContent = 'Requesting Camera...';
    const constraints = {
      video: {
        facingMode: state.facingMode,
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    };

    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = state.stream;
    
    await new Promise(resolve => {
      video.onloadedmetadata = () => {
        video.play();
        resolve();
      };
    });

    state.isCameraRunning = true;
    placeholder.style.display = 'none';
    laser.classList.add('scanning');
    cameraBadge.textContent = 'Live Scanning';
    cameraBadge.style.color = 'var(--accent-cyan)';
    document.getElementById('toggleCamText').textContent = 'Stop Camera';
    document.getElementById('toggleCamIcon').textContent = '⏹';

    applyMirrorStyle();
    startRecognitionLoop();
  } catch (err) {
    console.error("Camera access denied or failed:", err);
    cameraBadge.textContent = 'Camera Unavailable';
    alert("Camera permission denied or camera device in use. You can use 'Upload Test Photo' or test profiles below.");
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }
  if (state.recognitionInterval) {
    clearInterval(state.recognitionInterval);
    state.recognitionInterval = null;
  }
  state.isCameraRunning = false;
  placeholder.style.display = 'flex';
  laser.classList.remove('scanning');
  cameraBadge.textContent = 'Webcam Standby';
  cameraBadge.style.color = '';
  document.getElementById('toggleCamText').textContent = 'Start Camera';
  document.getElementById('toggleCamIcon').textContent = '▶';

  // Clear overlay canvas
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function applyMirrorStyle() {
  if (state.isMirrored) {
    video.style.transform = 'scaleX(-1)';
    canvas.style.transform = 'scaleX(-1)';
  } else {
    video.style.transform = 'scaleX(1)';
    canvas.style.transform = 'scaleX(1)';
  }
}

// ------------------ REAL-TIME FACE RECOGNITION LOOP ------------------
async function startRecognitionLoop() {
  if (!state.isCameraRunning) return;

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const displaySize = { width: canvas.width, height: canvas.height };
  faceapi.matchDimensions(canvas, displaySize);

  let isDetecting = false;

  state.recognitionInterval = setInterval(async () => {
    if (!state.isCameraRunning || isDetecting || !state.modelsLoaded) return;
    isDetecting = true;

    try {
      const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
      const detections = await faceapi
        .detectAllFaces(video, options)
        .withFaceLandmarks(true)
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (resizedDetections.length > 0 && state.faceMatcher) {
        resizedDetections.forEach(detection => {
          const match = state.faceMatcher.findBestMatch(detection.descriptor);
          const box = detection.detection.box;
          const confidence = Math.round((1 - match.distance) * 100);

          if (match.label !== 'unknown') {
            // Recognized Face -> Green HUD Box
            drawCyberpunkBox(ctx, box, match.label, `${confidence}% Match`, '#10b981');
            markAttendance(match.label, confidence);
          } else {
            // Unregistered Face -> Red HUD Box
            drawCyberpunkBox(ctx, box, 'VISITOR / NOT REGISTERED', 'Access Restricted', '#ef4444');
          }
        });
      }
    } catch (err) {
      console.warn("Detection cycle exception:", err);
    } finally {
      isDetecting = false;
    }
  }, 160); // ~6-7 FPS inference for silky smooth responsiveness without burning CPU
}

// Draw stylish bounding box with corner brackets and badges
function drawCyberpunkBox(ctx, box, title, subtitle, color) {
  const { x, y, width, height } = box;
  const bracketSize = Math.min(width, height) * 0.2;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;

  // Corner brackets
  ctx.beginPath();
  // Top-Left
  ctx.moveTo(x, y + bracketSize); ctx.lineTo(x, y); ctx.lineTo(x + bracketSize, y);
  // Top-Right
  ctx.moveTo(x + width - bracketSize, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y + bracketSize);
  // Bottom-Left
  ctx.moveTo(x, y + height - bracketSize); ctx.lineTo(x, y + height); ctx.lineTo(x + bracketSize, y + height);
  // Bottom-Right
  ctx.moveTo(x + width - bracketSize, y + height); ctx.lineTo(x + width, y + height); ctx.lineTo(x + width, y + height - bracketSize);
  ctx.stroke();

  // Subtle interior glow box
  ctx.fillStyle = color === '#10b981' ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)';
  ctx.fillRect(x, y, width, height);

  // Top Label Tag
  const tagHeight = 28;
  const tagWidth = Math.max(140, ctx.measureText(title).width + 24);
  const tagY = y > tagHeight + 6 ? y - tagHeight - 4 : y + height + 6;

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, tagY, tagWidth, tagHeight, 4) : ctx.rect(x, tagY, tagWidth, tagHeight);
  ctx.fill();

  ctx.fillStyle = '#050811';
  ctx.font = 'bold 13px Inter, sans-serif';
  ctx.fillText(title, x + 8, tagY + 18);

  ctx.restore();
}

// ------------------ TEST RECOGNITION BY IMAGE ------------------
async function recognizeFromImage(imgElement) {
  if (!state.modelsLoaded) {
    alert("AI Models are still initializing. Please wait 2 seconds.");
    return;
  }

  try {
    const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 });
    const detections = await faceapi
      .detectAllFaces(imgElement, options)
      .withFaceLandmarks(true)
      .withFaceDescriptors();

    if (detections.length === 0) {
      alert("No face detected in the selected image. Please try a clear frontal photo.");
      return;
    }

    let recognizedNames = [];
    detections.forEach(detection => {
      if (state.faceMatcher) {
        const match = state.faceMatcher.findBestMatch(detection.descriptor);
        const confidence = Math.round((1 - match.distance) * 100);
        if (match.label !== 'unknown') {
          markAttendance(match.label, confidence);
          recognizedNames.push(`${match.label} (${confidence}%)`);
        } else {
          showDetectionAlert("Unregistered Visitor", 0, false);
          recognizedNames.push("Unknown Visitor");
        }
      }
    });

    console.log("Photo Recognition Result:", recognizedNames.join(', '));
  } catch (err) {
    console.error("Image recognition failed:", err);
  }
}

// Trigger simulation for demo profile
window.triggerTestPerson = async function(name) {
  const profile = state.registeredProfiles.find(p => p.name.toLowerCase() === name.toLowerCase());
  if (!profile) return;

  const confidence = Math.floor(Math.random() * 6) + 93; // 93% - 98%
  markAttendance(profile.name, confidence);
};

// ------------------ REGISTER NEW MEMBER MODAL ------------------
const registerModal = document.getElementById('registerModal');
const enrollVideo = document.getElementById('enrollWebcamVideo');
const enrollImgPreview = document.getElementById('enrollImgPreview');
const enrollPlaceholder = document.getElementById('enrollPlaceholder');
const startEnrollCamBtn = document.getElementById('startEnrollCamBtn');
const captureEnrollSnapshotBtn = document.getElementById('captureEnrollSnapshotBtn');

function openRegisterModal() {
  document.getElementById('registerForm').reset();
  state.currentEnrollDescriptor = null;
  state.currentEnrollImageBase64 = null;
  enrollImgPreview.style.display = 'none';
  enrollVideo.style.display = 'none';
  enrollPlaceholder.style.display = 'flex';
  captureEnrollSnapshotBtn.style.display = 'none';
  registerModal.showModal();
}

function closeRegisterModal() {
  stopEnrollCamera();
  registerModal.close();
}

async function startEnrollCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: 'user' }
    });
    state.enrollStream = stream;
    enrollVideo.srcObject = stream;
    enrollVideo.style.display = 'block';
    enrollImgPreview.style.display = 'none';
    enrollPlaceholder.style.display = 'none';
    captureEnrollSnapshotBtn.style.display = 'inline-flex';
    startEnrollCamBtn.style.display = 'none';
  } catch (err) {
    alert("Could not access camera for enrollment: " + err.message);
  }
}

function stopEnrollCamera() {
  if (state.enrollStream) {
    state.enrollStream.getTracks().forEach(t => t.stop());
    state.enrollStream = null;
  }
  startEnrollCamBtn.style.display = 'inline-flex';
  captureEnrollSnapshotBtn.style.display = 'none';
}

async function captureEnrollSnapshot() {
  const c = document.createElement('canvas');
  c.width = enrollVideo.videoWidth || 320;
  c.height = enrollVideo.videoHeight || 240;
  const ctx = c.getContext('2d');
  ctx.drawImage(enrollVideo, 0, 0, c.width, c.height);

  const dataUrl = c.toDataURL('image/jpeg', 0.85);
  stopEnrollCamera();

  enrollImgPreview.src = dataUrl;
  enrollImgPreview.style.display = 'block';
  enrollVideo.style.display = 'none';
  state.currentEnrollImageBase64 = dataUrl;

  await computeAndSetEnrollDescriptor(enrollImgPreview);
}

async function computeAndSetEnrollDescriptor(imageOrVideoElement) {
  try {
    const detection = await faceapi
      .detectSingleFace(imageOrVideoElement, new faceapi.TinyFaceDetectorOptions({ inputSize: 224 }))
      .withFaceLandmarks(true)
      .withFaceDescriptor();

    if (!detection) {
      alert("No face detected in the portrait. Please ensure good lighting and face clearly visible.");
      state.currentEnrollDescriptor = null;
      return false;
    }

    state.currentEnrollDescriptor = detection.descriptor;
    console.log("[Enrollment] Face descriptor extracted successfully (128-d vector)");
    return true;
  } catch (err) {
    console.error("Descriptor computation failed:", err);
    return false;
  }
}

// ------------------ EXPORT TO CSV ------------------
function exportAttendanceCsv() {
  if (state.attendanceRecords.length === 0) {
    alert("No attendance records to export.");
    return;
  }

  const headers = ["Name", "Date", "Time", "Status", "Confidence"];
  const rows = state.attendanceRecords.map(r => [
    `"${r.name}"`,
    `"${r.date}"`,
    `"${r.time}"`,
    `"${r.status || 'Present'}"`,
    `"${r.confidence || 95}%"`
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `attendance_report_${new Date().toISOString().split('T')[0]}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ------------------ CLEAR ATTENDANCE LOGS ------------------
function clearAttendanceLogs() {
  if (!confirm("Are you sure you want to clear all attendance logs? This cannot be undone.")) return;
  state.attendanceRecords = [];
  localStorage.setItem('ATTENDANCE_LOGS', JSON.stringify([]));
  fetch('/api/attendance/clear', { method: 'POST' }).catch(() => {});
  renderAttendanceTable();
  updateStats();
}

// ------------------ EVENT LISTENERS & SETUP ------------------
function bindEventListeners() {
  // Start / Stop Camera
  document.getElementById('startCameraBtn').addEventListener('click', startCamera);
  document.getElementById('toggleCamBtn').addEventListener('click', () => {
    state.isCameraRunning ? stopCamera() : startCamera();
  });

  // Switch Camera Lens
  document.getElementById('switchCamBtn').addEventListener('click', () => {
    state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
    if (state.isCameraRunning) {
      stopCamera();
      startCamera();
    }
  });

  // Flip Mirror
  document.getElementById('flipMirrorBtn').addEventListener('click', () => {
    state.isMirrored = !state.isMirrored;
    applyMirrorStyle();
  });

  // Snapshot Manual Check-in
  document.getElementById('snapshotCheckinBtn').addEventListener('click', async () => {
    if (!state.isCameraRunning) {
      alert("Please start the camera first.");
      return;
    }
    const c = document.createElement('canvas');
    c.width = video.videoWidth || 640;
    c.height = video.videoHeight || 480;
    const ctx = c.getContext('2d');
    ctx.drawImage(video, 0, 0, c.width, c.height);
    const img = new Image();
    img.src = c.toDataURL('image/jpeg');
    img.onload = () => recognizeFromImage(img);
  });

  // Test photo upload input (viewfinder)
  document.getElementById('testPhotoInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target.result;
      img.onload = () => recognizeFromImage(img);
    };
    reader.readAsDataURL(file);
  });

  // Scan Photo Toolbar input
  document.getElementById('scanPhotoInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target.result;
      img.onload = () => recognizeFromImage(img);
    };
    reader.readAsDataURL(file);
  });

  // Filters
  document.getElementById('searchFilter').addEventListener('input', renderAttendanceTable);
  document.getElementById('dateFilter').addEventListener('change', renderAttendanceTable);

  // CSV Export & Clear
  document.getElementById('exportCsvBtn').addEventListener('click', exportAttendanceCsv);
  document.getElementById('clearLogsBtn').addEventListener('click', clearAttendanceLogs);

  // Modals
  document.getElementById('openRegisterModalBtn').addEventListener('click', openRegisterModal);
  document.getElementById('openRegisterModalBtn2').addEventListener('click', openRegisterModal);
  document.getElementById('closeRegisterModalBtn').addEventListener('click', closeRegisterModal);
  document.getElementById('cancelRegisterModalBtn').addEventListener('click', closeRegisterModal);

  // Modal Backdrop click to close
  registerModal.addEventListener('click', (e) => {
    if (e.target === registerModal) closeRegisterModal();
  });

  // Enroll Camera Actions
  startEnrollCamBtn.addEventListener('click', startEnrollCamera);
  captureEnrollSnapshotBtn.addEventListener('click', captureEnrollSnapshot);

  // Enroll file upload
  document.getElementById('enrollFileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    stopEnrollCamera();
    const reader = new FileReader();
    reader.onload = async (event) => {
      enrollImgPreview.src = event.target.result;
      enrollImgPreview.style.display = 'block';
      enrollPlaceholder.style.display = 'none';
      state.currentEnrollImageBase64 = event.target.result;
      enrollImgPreview.onload = () => computeAndSetEnrollDescriptor(enrollImgPreview);
    };
    reader.readAsDataURL(file);
  });

  // Register Form Submit
  document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('newMemberName').value.trim();
    const role = document.getElementById('newMemberRole').value.trim() || 'Member';
    const id = document.getElementById('newMemberId').value.trim() || `ID-${Date.now().toString().slice(-4)}`;

    if (!name) {
      alert("Please provide the full name.");
      return;
    }

    if (!state.currentEnrollDescriptor) {
      alert("Please capture or upload a clear facial portrait for enrollment.");
      return;
    }

    const newProfile = {
      name: name,
      role: role,
      id: id,
      image: state.currentEnrollImageBase64 || 'images/ipsita.jpeg',
      descriptor: Array.from(state.currentEnrollDescriptor)
    };

    // Save to custom faces in localStorage
    try {
      const raw = localStorage.getItem('CUSTOM_FACES') || '[]';
      const list = JSON.parse(raw);
      list.push(newProfile);
      localStorage.setItem('CUSTOM_FACES', JSON.stringify(list));
    } catch (err) {
      console.error("Storage error:", err);
    }

    // Update in-memory profile
    state.registeredProfiles.push({
      ...newProfile,
      descriptor: state.currentEnrollDescriptor
    });

    updateFaceMatcher();
    renderPersonnelDirectory();
    renderDemoButtons();
    updateStats();
    closeRegisterModal();

    alert(`Successfully registered ${name} in the biometric database!`);
  });
}

// ------------------ INITIALIZE FACE-API NEURAL MODELS ------------------
async function initFaceApi() {
  const chip = document.getElementById('systemStatusChip');
  const chipText = document.getElementById('systemStatusText');

  chipText.textContent = 'Loading Neural Weights...';
  chip.classList.remove('ready');

  // Multi-CDN fallback for model weights
  const MODEL_URLS = [
    'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/',
    'https://raw.githubusercontent.com/vladmandic/face-api/master/model/'
  ];

  let loaded = false;
  for (const url of MODEL_URLS) {
    try {
      console.log(`[Neural Engine] Fetching models from ${url}...`);
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(url),
        faceapi.nets.faceLandmark68Net.loadFromUri(url),
        faceapi.nets.faceRecognitionNet.loadFromUri(url)
      ]);
      loaded = true;
      console.log("[Neural Engine] Models loaded successfully!");
      break;
    } catch (err) {
      console.warn(`Failed loading models from ${url}:`, err);
    }
  }

  state.modelsLoaded = loaded;
  if (loaded) {
    chipText.textContent = 'Neural AI Ready';
    chip.classList.add('ready');
    updateFaceMatcher();
  } else {
    chipText.textContent = 'Engine Offline (Using Cache)';
    console.warn("Neural models couldn't be loaded from CDN. Preloaded descriptors still active.");
  }
}

// ------------------ APP BOOTSTRAP ------------------
window.addEventListener('DOMContentLoaded', async () => {
  initLiveClock();
  loadAttendanceLogs();
  loadProfiles();
  bindEventListeners();
  await initFaceApi();
});
