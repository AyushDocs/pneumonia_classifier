let selectedFile = null;
let lastResult = null; // Store last result for report generation

const patientIdInp = document.getElementById('patient-id');
const fileInp = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const fileNameText = document.getElementById('file-name');
const predictBtn = document.getElementById('predict-btn');
const welcome = document.getElementById('welcome-message');
const result = document.getElementById('result-display');
const loader = document.getElementById('loader');
const historyList = document.getElementById('history-list');
const downloadBtn = document.getElementById('download-report-btn');
const themeToggleBtn = document.getElementById('theme-toggle');
const opacitySlider = document.getElementById('heatmap-opacity');
const opacityValue = document.getElementById('opacity-value');
const opacityControl = document.getElementById('opacity-control');

// Auth DOM
const loginModal = document.getElementById('login-modal');
const mainApp = document.getElementById('main-app');
const loginUsernameInp = document.getElementById('login-username');
const loginPasswordInp = document.getElementById('login-password');
const loginSubmitBtn = document.getElementById('login-submit-btn');
const loginError = document.getElementById('login-error');

function getAuthHeaders() {
    const token = localStorage.getItem('jwt');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

function requireAuth() {
    const token = localStorage.getItem('jwt');
    if (!token) {
        loginModal.style.display = 'flex';
        mainApp.style.display = 'none';
        return false;
    } else {
        loginModal.style.display = 'none';
        mainApp.style.display = 'flex';

        // Auto-fill Doctor ID if role is doctor
        const role = localStorage.getItem('role');
        const username = localStorage.getItem('username');
        if (role === 'Doctor') {
            document.getElementById('doctor-id').value = username || "dr_smith";
        }
        return true;
    }
}

loginSubmitBtn.addEventListener('click', async () => {
    loginError.style.display = 'none';
    loginSubmitBtn.innerText = "Authenticating...";
    loginSubmitBtn.disabled = true;

    try {
        const params = new URLSearchParams();
        params.append('username', loginUsernameInp.value);
        params.append('password', loginPasswordInp.value);

        const response = await fetch('/token', {
            method: 'POST',
            body: params
        });

        if (!response.ok) throw new Error("Invalid username or password");

        const data = await response.json();
        localStorage.setItem('jwt', data.access_token);
        localStorage.setItem('role', data.role);
        localStorage.setItem('username', loginUsernameInp.value);

        requireAuth();
    } catch(err) {
        loginError.innerText = err.message;
        loginError.style.display = 'block';
    } finally {
        loginSubmitBtn.innerText = "Authenticate";
        loginSubmitBtn.disabled = false;
    }
});

// Run auth check early
requireAuth();

let viewer = null;
const worker = new Worker('/static/worker.js');

// Dark Mode Logic
if (themeToggleBtn) {
    // Check local storage or system preference
    if (localStorage.getItem('theme') === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.body.classList.add('dark-theme');
        themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
    }

    themeToggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-theme');
        if (document.body.classList.contains('dark-theme')) {
            localStorage.setItem('theme', 'dark');
            themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            localStorage.setItem('theme', 'light');
            themeToggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
        }
    });
}

// Handle File Selection
dropZone.addEventListener('click', () => fileInp.click());

fileInp.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        fileNameText.innerText = `Selected: ${file.name}`;
        fileNameText.classList.add('text-primary');
    }
});

const sampleBtn = document.getElementById('try-sample-btn');
if (sampleBtn) {
    sampleBtn.addEventListener('click', async () => {
        try {
            sampleBtn.disabled = true;
            sampleBtn.style.opacity = '0.5';
            const response = await fetch('/static/sample.jpg');
            if(!response.ok) throw new Error("Sample missing");
            const blob = await response.blob();
            const file = new File([blob], "sample_xray.jpg", { type: "image/jpeg" });
            selectedFile = file;
            fileNameText.innerText = `Selected: sample_xray.jpg`;
            fileNameText.classList.add('text-primary');
            patientIdInp.value = "PX-SAMPLE";
            patientIdInp.classList.remove('border-error');
        } catch(e) {
            alert("Could not load sample image.");
        } finally {
            sampleBtn.disabled = false;
            sampleBtn.style.opacity = '1';
        }
    });
}

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-primary');
    dropZone.classList.remove('border-muted');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.add('border-muted');
    dropZone.classList.remove('border-primary');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-muted');
    dropZone.classList.remove('border-primary');
    const file = e.dataTransfer.files[0];
    if (file) {
        selectedFile = file;
        fileNameText.innerText = `Selected: ${file.name}`;
        fileNameText.classList.add('text-primary');
    }
});

// Fetch History
async function updateHistory() {
    const patientId = patientIdInp.value.trim() || "DEFAULT";
    try {
        const response = await fetch(`/history/${patientId}`, {
            headers: getAuthHeaders()
        });
        if (response.status === 401 || response.status === 403) {
            localStorage.removeItem('jwt');
            requireAuth();
            return;
        }
        const data = await response.json();
        if (data.status === "success" && data.history.length > 0) {
            historyList.innerHTML = data.history.map(item => `
                <div class="history-item">
                    <span class="${item[0] === 'Pneumonia' ? 'status-pneumonia' : 'status-normal'}">${item[0]}</span>
                    <div class="meta">${item[2]}</div>
                </div>
            `).join('');
        } else {
            historyList.innerHTML = '<p class="history-empty-text">No history for this ID.</p>';
        }
    } catch (err) {
        console.error("History fetch error:", err);
    }
}

// Predict
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert("Please select an X-ray image first.");
        return;
    }

    const patientId = patientIdInp.value.trim() || "DEFAULT";
    const doctorId = document.getElementById('doctor-id').value.trim() || "UNKNOWN";

    welcome.style.display = 'none';
    result.style.display = 'none';
    loader.style.display = 'block';
    predictBtn.disabled = true;

    // Run Frontend OOD Check
    await new Promise((resolve) => {
        worker.onmessage = function(e) {
            if (!e.data.valid) {
                loader.style.display = 'none';
                welcome.style.display = 'block';
                welcome.innerHTML = `<div class="error-message"><i class="fas fa-exclamation-triangle"></i><br>${e.data.reason}</div>`;
                predictBtn.disabled = false;
                resolve(false);
            } else {
                resolve(true); // Is valid
            }
        };
        worker.postMessage(selectedFile);
    }).then(async (proceed) => {
        if (!proceed) return;

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('patient_id', patientId);
            formData.append('requester_id', doctorId);

            const response = await fetch('/predict', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: formData
            });

            if (response.status === 401 || response.status === 403) {
                 localStorage.removeItem('jwt');
                 requireAuth();
                 throw new Error("Session expired or insufficient Doctor privileges.");
            }

            const queueData = await response.json();

            if (!response.ok || queueData.status === 'error') {
                throw new Error(queueData.message || `Server returned ${response.status}`);
            }

            const jobId = queueData.job_id;

            // Polling function
            const pollInterval = setInterval(async () => {
                try {
                    const statusRes = await fetch(`/predict/status/${jobId}`, {
                        headers: getAuthHeaders()
                    });
                    const data = await statusRes.json();

                    if (data.status === 'processing') {
                        // Still waiting
                        return;
                    }

                    // Done polling
                    clearInterval(pollInterval);
                    loader.style.display = 'none';

                    if (data.status === "success") {
                        result.style.display = 'block';
                        lastResult = { ...data, patient_id: patientId };

                        const badge = document.getElementById('prediction-badge');
                        badge.innerText = data.prediction.toUpperCase();
                        badge.className = 'prediction-badge-new ' + (data.prediction === 'Pneumonia' ? 'bg-pneumonia' : 'bg-normal');

                        document.getElementById('confidence-val').innerText = data.confidence;
                        document.getElementById('message').innerText = data.message;
                        downloadBtn.style.display = 'block';

                        // Interactive Layered Heatmap
                        if (data.heatmap) {
                            opacityControl.style.display = 'block';
                            const baseImageURL = URL.createObjectURL(selectedFile);
                            if (viewer) viewer.destroy();

                            viewer = OpenSeadragon({
                                id: "osd-viewer",
                                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
                                tileSources: [
                                    { type: 'image', url: baseImageURL },
                                    { type: 'image', url: data.heatmap, opacity: parseFloat(opacitySlider.value) }
                                ],
                                showNavigationControl: false,
                                gestureSettingsMouse: { clickToZoom: true }
                            });

                            opacitySlider.oninput = (e) => {
                                const val = e.target.value;
                                opacityValue.innerText = Math.round(val * 100) + '%';
                                if (viewer && viewer.world.getItemCount() > 1) {
                                    viewer.world.getItemAt(1).setOpacity(parseFloat(val));
                                }
                            };
                        } else {
                            opacityControl.style.display = 'none';
                        }

                        updateHistory();
                    } else {
                        welcome.style.display = 'block';
                        welcome.innerHTML = `<div class="error-message"><i class="fas fa-exclamation-triangle"></i><br>${data.message}</div>`;
                    }

                    predictBtn.disabled = false;

                } catch (pollErr) {
                    clearInterval(pollInterval);
                    console.error('Polling error:', pollErr);
                    loader.style.display = 'none';
                    welcome.style.display = 'block';
                    welcome.innerHTML = `<div class="error-message">Network anomaly while waiting: ${pollErr.message}</div>`;
                    predictBtn.disabled = false;
                }
            }, 1000); // Check every 1 second

        } catch (err) {
            console.error(err);
            loader.style.display = 'none';
            welcome.style.display = 'block';
            welcome.innerHTML = `<div class="error-message">Network anomaly: ${err.message}. Is the server running?</div>`;
            predictBtn.disabled = false;
        }
    }); // End of promise chain from WebWorker
});

// Download Report
downloadBtn.addEventListener('click', async () => {
    if (!lastResult) return;

    downloadBtn.disabled = true;
    downloadBtn.innerText = "Generating Report...";

    try {
        const response = await fetch('/generate_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify({
                patient_id: lastResult.patient_id,
                prediction: lastResult.prediction,
                confidence: lastResult.confidence,
                heatmap: lastResult.heatmap
            })
        });

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Clinical_Report_${lastResult.patient_id}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        console.error("Report error:", err);
        alert("Could not generate report.");
    } finally {
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = '<i class="fas fa-file-pdf"></i> Download Clinical Report (PDF)';
    }
});

// Load history on ID change
patientIdInp.addEventListener('blur', updateHistory);
