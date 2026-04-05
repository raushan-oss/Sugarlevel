document.addEventListener("DOMContentLoaded", () => {
    
    // --- Index Page Logic ---
    const uploadForm = document.getElementById("uploadForm");
    if (uploadForm) {
        uploadForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById("csvFile");
            const statusDiv = document.getElementById("uploadStatus");
            const btn = uploadForm.querySelector("button");
            
            if (fileInput.files.length === 0) {
                statusDiv.innerText = "Please select a file.";
                statusDiv.className = "status-msg error";
                return;
            }
            
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append("file", file);
            
            btn.innerText = "Uploading...";
            btn.disabled = true;
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                if (response.ok) {
                    statusDiv.innerText = result.success;
                    statusDiv.className = "status-msg success";
                } else {
                    statusDiv.innerText = result.error || "Upload failed.";
                    statusDiv.className = "status-msg error";
                }
            } catch (error) {
                statusDiv.innerText = "Network error. Please try again.";
                statusDiv.className = "status-msg error";
            } finally {
                btn.innerText = "Upload & Train";
                btn.disabled = false;
            }
        });
    }

    // --- Dashboard Page Logic ---
    const predictionForm = document.getElementById("predictionForm");
    if (predictionForm) {
        // Set default timestamp to now
        const now = new Date();
        now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
        document.getElementById("timestamp").value = now.toISOString().slice(0,16);

        // Initialize Chart
        const ctx = document.getElementById('glucoseChart').getContext('2d');
        const glucoseChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], // Timestamps
                datasets: [{
                    label: 'Predicted Glucose Trend (Simulated)',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Hypoglycemia Threshold',
                    data: [], // Will be filled with 70s
                    borderColor: '#ef4444',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                }
            }
        });

        predictionForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            const btn = predictionForm.querySelector("button");
            btn.innerText = "Predicting...";
            btn.disabled = true;

            const payload = {
                timestamp: document.getElementById("timestamp").value,
                glucose: parseFloat(document.getElementById("glucose").value),
                carbs: parseFloat(document.getElementById("carbs").value),
                insulin: parseFloat(document.getElementById("insulin").value),
                activity: document.getElementById("activity").value,
                sleep: parseFloat(document.getElementById("sleep").value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Update UI
                    document.getElementById("resRisk").innerText = result.risk_level;
                    document.getElementById("resProb").innerText = result.probability + "%";
                    document.getElementById("resExplanation").innerText = result.explanation;
                    
                    const alertBox = document.getElementById("alertBox");
                    if (result.risk_level === 'High') {
                        alertBox.classList.remove("hidden");
                        document.getElementById("resRisk").style.color = "var(--danger)";
                    } else if (result.risk_level === 'Medium') {
                        alertBox.classList.add("hidden");
                        document.getElementById("resRisk").style.color = "var(--warning)";
                    } else {
                        alertBox.classList.add("hidden");
                        document.getElementById("resRisk").style.color = "var(--success)";
                    }

                    // Update Chart Simulation
                    // To make it look cool, we simulate the drop based on the risk level
                    const timeLabels = ['Now', '+10m', '+20m', '+30m'];
                    let currentG = payload.glucose;
                    let trend;
                    
                    if (result.risk_level === 'High') {
                        trend = [currentG, currentG - 10, currentG - 25, currentG - 40];
                    } else if (result.risk_level === 'Medium') {
                        trend = [currentG, currentG - 5, currentG - 10, currentG - 15];
                    } else {
                        trend = [currentG, currentG + 2, currentG + 5, currentG + 3];
                    }
                    
                    glucoseChart.data.labels = timeLabels;
                    glucoseChart.data.datasets[0].data = trend;
                    glucoseChart.data.datasets[1].data = [70, 70, 70, 70];
                    glucoseChart.update();
                    
                } else {
                    alert("Error: " + (result.error || "Unknown error"));
                }
            } catch (error) {
                alert("Network error. Please make sure the backend is running.");
            } finally {
                btn.innerText = "Predict Risk";
                btn.disabled = false;
            }
        });
    }
});
