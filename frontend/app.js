/**
 * MedAssist-AI: Clean Frontend Application Controller
 * Handles real image uploads, drag-and-drop, multimodal fusion, and uncluttered display.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentCaseId = null;
  let currentFile = null;
  let analysisData = null;
  let activeTabMode = "heatmap"; // 'heatmap' | 'contour' | 'original'
  let currentZoom = 1.0;
  let selectedPathology = null;

  // DOM Elements
  const globalFileInput = document.getElementById("globalFileInput");
  const btnHeaderUpload = document.getElementById("btnHeaderUpload");
  const presetSelect = document.getElementById("presetSelect");
  const hwTag = document.getElementById("hwTag");
  const hwText = document.getElementById("hwText");

  // Viewer Elements
  const tabHeatmap = document.getElementById("tabHeatmap");
  const tabContour = document.getElementById("tabContour");
  const tabOriginal = document.getElementById("tabOriginal");
  const allTabs = [tabHeatmap, tabContour, tabOriginal];
  
  const colormapSelect = document.getElementById("colormapSelect");
  const qualityBadge = document.getElementById("qualityBadge");
  const qualityDot = document.getElementById("qualityDot");
  const qualityText = document.getElementById("qualityText");
  const viewportDropzone = document.getElementById("viewportDropzone");
  const mainXray = document.getElementById("mainXray");
  const dropHint = document.getElementById("dropHint");

  const opacitySlider = document.getElementById("opacitySlider");
  const opacityLabel = document.getElementById("opacityLabel");
  const btnZoomIn = document.getElementById("btnZoomIn");
  const btnZoomOut = document.getElementById("btnZoomOut");
  const btnZoomReset = document.getElementById("btnZoomReset");

  // Notes & Action
  const clinicalNotes = document.getElementById("clinicalNotes");
  const btnClearNotes = document.getElementById("btnClearNotes");
  const snippetButtons = document.querySelectorAll(".btn-snippet");
  const btnRunAnalysis = document.getElementById("btnRunAnalysis");
  const btnExportReport = document.getElementById("btnExportReport");

  // Results Elements
  const loadingState = document.getElementById("loadingState");
  const guardPillBar = document.getElementById("guardPillBar");
  const badgeVerified = document.getElementById("badgeVerified");
  const badgeBlocked = document.getElementById("badgeBlocked");
  const findingsFeed = document.getElementById("findingsFeed");
  const doctorBox = document.getElementById("doctorBox");
  const doctorBoxBody = document.getElementById("doctorBoxBody");

  const suppressedBox = document.getElementById("suppressedBox");
  const suppressedToggle = document.getElementById("suppressedToggle");
  const suppressedList = document.getElementById("suppressedList");
  const suppressedCount = document.getElementById("suppressedCount");
  const suppressedArrow = document.getElementById("suppressedArrow");

  // Modal Elements
  const reportModal = document.getElementById("reportModal");
  const modalReportContent = document.getElementById("modalReportContent");
  const btnCloseModal = document.getElementById("btnCloseModal");
  const btnDismissModal = document.getElementById("btnDismissModal");

  // ==========================================================================
  // Initialization
  // ==========================================================================

  init();

  async function init() {
    setupEvents();
    await checkHardware();
    await loadPresets();
  }

  async function checkHardware() {
    try {
      const res = await fetch("/api/system/status");
      const data = await res.json();
      if (data.device === "cuda") {
        hwText.textContent = `${data.gpu_name} (Active)`;
      } else {
        hwText.textContent = "Ryzen 7 (CPU Fallback)";
      }
    } catch (e) {
      console.warn("Backend status unavailable:", e);
    }
  }

  async function loadPresets() {
    try {
      const res = await fetch("/api/cases");
      const data = await res.json();
      presetSelect.innerHTML = "";
      
      data.cases.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.id;
        opt.textContent = `${c.title}`;
        presetSelect.appendChild(opt);
      });

      if (data.cases.length > 0) {
        presetSelect.value = data.cases[0].id;
        loadPresetCase(data.cases[0].id);
      }
    } catch (e) {
      console.error("Failed to load cases:", e);
    }
  }

  function loadPresetCase(caseId) {
    currentCaseId = caseId;
    currentFile = null;
    dropHint.classList.add("hidden");

    fetch("/api/cases")
      .then((r) => r.json())
      .then((d) => {
        const c = d.cases.find((item) => item.id === caseId);
        if (c) {
          clinicalNotes.value = c.clinical_notes;
        }
      });

    const imgUrl = `/api/cases/${caseId}/image?t=${Date.now()}`;
    mainXray.src = imgUrl;

    executeAnalysis();
  }

  // ==========================================================================
  // Event Bindings
  // ==========================================================================

  function setupEvents() {
    // Header upload trigger
    btnHeaderUpload.addEventListener("click", () => globalFileInput.click());
    viewportDropzone.addEventListener("click", (e) => {
      if (dropHint.contains(e.target) || e.target === viewportDropzone) {
        globalFileInput.click();
      }
    });

    globalFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleUploadedFile(e.target.files[0]);
      }
    });

    // Drag and drop on viewport
    viewportDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      viewportDropzone.style.borderColor = "#0ea5e9";
    });

    viewportDropzone.addEventListener("dragleave", () => {
      viewportDropzone.style.borderColor = "";
    });

    viewportDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      viewportDropzone.style.borderColor = "";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleUploadedFile(e.dataTransfer.files[0]);
      }
    });

    // Preset dropdown
    presetSelect.addEventListener("change", (e) => {
      loadPresetCase(e.target.value);
    });

    // View tabs
    allTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        allTabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        activeTabMode = tab.dataset.mode;
        renderActiveView();
      });
    });

    // Colormap
    colormapSelect.addEventListener("change", () => {
      if (analysisData) executeAnalysis();
    });

    // Slider
    opacitySlider.addEventListener("input", (e) => {
      const v = e.target.value;
      opacityLabel.textContent = `${v}%`;
      if (activeTabMode === "heatmap") {
        mainXray.style.opacity = Math.max(0.3, v / 100);
      }
    });

    // Zoom
    btnZoomIn.addEventListener("click", () => {
      currentZoom = Math.min(2.5, currentZoom + 0.2);
      mainXray.style.transform = `scale(${currentZoom})`;
    });
    btnZoomOut.addEventListener("click", () => {
      currentZoom = Math.max(0.6, currentZoom - 0.2);
      mainXray.style.transform = `scale(${currentZoom})`;
    });
    btnZoomReset.addEventListener("click", () => {
      currentZoom = 1.0;
      mainXray.style.transform = "scale(1)";
    });

    // Snippet buttons for quick patient notes
    snippetButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = btn.dataset.preset;
        if (p === "pneumonia") {
          clinicalNotes.value = "Patient presents with 3-day high fever (39.2°C), chills, productive cough with purulent yellow sputum, and pleuritic chest discomfort. WBC count: 16,400/mcL (leukocytosis). SpO2: 91%. Coarse inspiratory crackles over lower lung zone.";
        } else if (p === "chf") {
          clinicalNotes.value = "68 y/o with orthopnea, paroxysmal nocturnal dyspnea, and progressive ankle edema. BP: 164/94 mmHg, HR: 88 bpm. NT-proBNP elevated at 2,450 pg/mL. S3 gallop audible at apex.";
        } else if (p === "pneumothorax") {
          clinicalNotes.value = "24 y/o male with sudden acute left pleuritic chest pain and shortness of breath. SpO2: 89%, HR: 114 bpm (tachycardia). Decreased breath sounds on left side with hyperresonance.";
        }
      });
    });

    btnClearNotes.addEventListener("click", () => {
      clinicalNotes.value = "";
    });

    // Run Analysis
    btnRunAnalysis.addEventListener("click", executeAnalysis);

    // Suppressed Accordion
    suppressedToggle.addEventListener("click", () => {
      suppressedList.classList.toggle("hidden");
      suppressedArrow.textContent = suppressedList.classList.contains("hidden") ? "▼" : "▲";
    });

    // Export Report
    btnExportReport.addEventListener("click", showReportModal);
    btnCloseModal.addEventListener("click", () => reportModal.classList.add("hidden"));
    btnDismissModal.addEventListener("click", () => reportModal.classList.add("hidden"));
  }

  // ==========================================================================
  // File Upload Logic
  // ==========================================================================

  function handleUploadedFile(file) {
    currentFile = file;
    currentCaseId = null;
    dropHint.classList.add("hidden");

    // Display image in viewer
    const reader = new FileReader();
    reader.onload = (e) => {
      mainXray.src = e.target.result;
      executeAnalysis();
    };
    reader.readAsDataURL(file);
  }

  // ==========================================================================
  // Analysis Pipeline Execution
  // ==========================================================================

  async function executeAnalysis() {
    findingsFeed.innerHTML = "";
    guardPillBar.classList.add("hidden");
    doctorBox.classList.add("hidden");
    suppressedBox.classList.add("hidden");
    loadingState.classList.remove("hidden");

    const formData = new FormData();
    if (currentFile) {
      formData.append("image", currentFile);
    } else if (currentCaseId) {
      formData.append("preset_case_id", currentCaseId);
    } else {
      loadingState.classList.add("hidden");
      return;
    }

    formData.append("clinical_notes", clinicalNotes.value || "");
    formData.append("colormap_name", colormapSelect.value || "jet");
    if (selectedPathology) {
      formData.append("focus_pathology", selectedPathology);
    }

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error("Analysis failed on server");

      const data = await res.json();
      analysisData = data;

      // Update quality pill
      const iqa = data.image_quality;
      qualityText.textContent = `Quality: ${iqa.quality_score}% (${iqa.status})`;
      qualityDot.style.background = iqa.status === "DEGRADED" ? "#ef4444" : "#10b981";

      // Render view
      renderActiveView();

      // Render Findings
      renderFindings(data);

    } catch (err) {
      console.error(err);
      findingsFeed.innerHTML = `<div style="color: #ef4444; padding: 12px;">Assessment Error: ${err.message}</div>`;
    } finally {
      loadingState.classList.add("hidden");
    }
  }

  // ==========================================================================
  // View Rendering
  // ==========================================================================

  function renderActiveView() {
    if (!analysisData) return;
    const vis = analysisData.visualizations;
    const target = selectedPathology || analysisData.target_pathology;

    if (activeTabMode === "heatmap") {
      mainXray.src = vis.heatmaps[target] || vis.current_target_heatmap || vis.original_image;
      mainXray.style.opacity = Math.max(0.3, opacitySlider.value / 100);
    } else if (activeTabMode === "contour") {
      mainXray.src = vis.contours_and_boxes[target] || vis.current_target_contour || vis.original_image;
      mainXray.style.opacity = "1";
    } else {
      mainXray.src = vis.original_image;
      mainXray.style.opacity = "1";
    }
  }

  // ==========================================================================
  // Findings Rendering (Clean Cards)
  // ==========================================================================

  function renderFindings(data) {
    const validated = data.multimodal_findings || [];
    const rejected = data.rejected_hallucinations || [];

    // Guard Pill Bar
    badgeVerified.textContent = `${validated.length} Verified by Evidence`;
    badgeBlocked.textContent = `${rejected.length} Blocked without Proof`;
    guardPillBar.classList.remove("hidden");

    // Findings List
    findingsFeed.innerHTML = "";
    if (validated.length === 0) {
      findingsFeed.innerHTML = `<div style="color: #94a3b8; padding: 16px;">No acute abnormality detected above threshold. Normal scan baseline.</div>`;
    } else {
      validated.forEach((f, idx) => {
        const card = document.createElement("div");
        card.className = "clean-finding-card";
        if (idx === 0) card.style.borderColor = "#0ea5e9";

        const zone = f.visual_evidence.anatomical_zone;
        const citations = f.clinical_evidence.matched_citations || [];

        card.innerHTML = `
          <div class="card-title-row">
            <span class="finding-title">🩺 ${f.pathology}</span>
            <span class="finding-conf">${f.confidence_percent}% Confidence</span>
          </div>

          <div class="assistive-quote">
            ${f.assistive_recommendation}
          </div>

          <div class="card-badges">
            <span class="badge-tag highlight">📍 Zone: ${zone}</span>
            <span class="badge-tag ${citations.length > 0 ? 'highlight' : ''}">
              ${citations.length > 0 ? '📝 Chart Corroborated' : '🔍 Unreferenced in Chart'}
            </span>
            <span class="badge-tag">ROI: ${f.visual_evidence.area_percent}% Area</span>
          </div>
        `;

        card.addEventListener("click", () => {
          document.querySelectorAll(".clean-finding-card").forEach(c => c.style.borderColor = "");
          card.style.borderColor = "#0ea5e9";
          selectedPathology = f.pathology;
          renderActiveView();
        });

        findingsFeed.appendChild(card);
      });
    }

    // Doctor Box
    if (data.doctor_summary) {
      doctorBoxBody.textContent = data.doctor_summary;
      doctorBox.classList.remove("hidden");
    }

    // Suppressed Drawer
    if (rejected.length > 0) {
      suppressedCount.textContent = rejected.length;
      suppressedList.innerHTML = rejected.map(r => `
        <div style="padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
          <strong>${r.pathology}</strong>: ${r.rejection_reason}
        </div>
      `).join("");
      suppressedBox.classList.remove("hidden");
    }
  }

  // ==========================================================================
  // Report Modal
  // ==========================================================================

  function showReportModal() {
    if (!analysisData) return alert("Please run an analysis first.");
    const d = analysisData;
    const validated = d.multimodal_findings || [];

    modalReportContent.innerHTML = `
      <div style="background: #fff; color: #111; padding: 20px; font-family: sans-serif; border-radius: 8px;">
        <h2 style="color: #0284c7; margin-bottom: 4px;">MEDASSIST-AI CLINICAL CONSULTATION REPORT</h2>
        <p style="font-size: 0.8rem; color: #666; margin-bottom: 16px;">Doctor-Assistive Multimodal Second Opinion · Anti-Hallucination Verified</p>
        
        <div style="background: #f1f5f9; padding: 10px; border-radius: 6px; margin-bottom: 16px; font-size: 0.85rem;">
          <strong>Image Quality Audit:</strong> Status ${d.image_quality.status} (${d.image_quality.quality_score}/100) | Sharpness: ${d.image_quality.sharpness_index}
        </div>

        <h3 style="font-size: 0.95rem; margin-bottom: 8px;">Validated Radiological Findings:</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-bottom: 16px;">
          <thead>
            <tr style="background: #e2e8f0; text-align: left;">
              <th style="padding: 6px;">Finding</th>
              <th style="padding: 6px;">Confidence</th>
              <th style="padding: 6px;">Location</th>
              <th style="padding: 6px;">Doctor-Assistive Recommendation</th>
            </tr>
          </thead>
          <tbody>
            ${validated.map(f => `
              <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 6px;"><strong>${f.pathology}</strong></td>
                <td style="padding: 6px;">${f.confidence_percent}%</td>
                <td style="padding: 6px;">${f.visual_evidence.anatomical_zone}</td>
                <td style="padding: 6px; font-style: italic;">"${f.assistive_recommendation}"</td>
              </tr>
            `).join("")}
          </tbody>
        </table>

        <div style="background: #e0f2fe; padding: 12px; border-left: 4px solid #0284c7; border-radius: 4px; font-size: 0.85rem;">
          <strong>Attending Physician Summary:</strong><br/>
          ${d.doctor_summary.replace(/\n/g, '<br/>')}
        </div>
      </div>
    `;

    reportModal.classList.remove("hidden");
  }

});
