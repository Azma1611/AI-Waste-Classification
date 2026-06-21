/**
 * app.js — Frontend Application Logic
 * AI Waste Classification & Recycling Recommendation System
 *
 * Communicates with Flask backend POST /predict endpoint.
 *
 * Modules:
 *  1. Particle Background
 *  2. Image Handler (drag-drop, upload)
 *  3. API Client (Flask backend)
 *  4. UI Controller (DOM, animations, charts)
 *  5. History Manager (localStorage)
 *  6. Eco Ticker
 *  7. Main App Orchestrator
 */

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS & CONFIG
// ═══════════════════════════════════════════════════════════════════

const API_BASE = window.location.origin;
const PREDICT_URL = `${API_BASE}/predict`;
const HEALTH_URL = `${API_BASE}/health`;

// Category display metadata
const CATEGORY_META = {
  Hazardous: {
    icon: "☣️",
    emoji: "⚠️",
    color: "#ef233c",
    gradient: "linear-gradient(135deg, #d90429, #ef233c)",
    description:
      "Chemicals, batteries, paint, medical waste, and toxic materials that require special disposal.",
  },
  Recyclable: {
    icon: "♻️",
    emoji: "🔄",
    color: "#00b4d8",
    gradient: "linear-gradient(135deg, #0077b6, #00b4d8)",
    description:
      "Plastic bottles, glass jars, metal cans, paper, and cardboard that can be recycled.",
  },
  "Non-Recyclable": {
    icon: "🗑️",
    emoji: "❌",
    color: "#6c757d",
    gradient: "linear-gradient(135deg, #495057, #6c757d)",
    description:
      "Items that don't fit standard recycling categories and go to general waste.",
  },
  Organic: {
    icon: "🌿",
    emoji: "🍂",
    color: "#00d4aa",
    gradient: "linear-gradient(135deg, #00a884, #00d4aa)",
    description:
      "Food waste, yard trimmings, plants, and biodegradable materials for composting.",
  },
};

// Eco tips for rotating ticker
const ECO_TIPS = [
  {
    tip: "The average person generates about 4.5 pounds of waste per day.",
    icon: "🌍",
  },
  { tip: "Only 9% of all plastic ever produced has been recycled.", icon: "♻️" },
  {
    tip: "Recycling one aluminum can saves enough energy to listen to a full album.",
    icon: "🥫",
  },
  {
    tip: "Glass bottles take 1 million years to decompose in a landfill.",
    icon: "🫙",
  },
  {
    tip: "Composting can reduce household waste by up to 30%.",
    icon: "🌿",
  },
  {
    tip: "E-waste represents only 2% of trash but 70% of toxic waste.",
    icon: "💻",
  },
  {
    tip: "Improperly disposed batteries can leak toxic chemicals into soil.",
    icon: "⚠️",
  },
  {
    tip: "Buying in bulk can reduce packaging waste by up to 50%.",
    icon: "📦",
  },
  {
    tip: "The Great Pacific Garbage Patch is twice the size of Texas.",
    icon: "🌊",
  },
  {
    tip: "Recycling steel saves 60% of the energy needed to make new steel.",
    icon: "🔩",
  },
  {
    tip: "Food waste in landfills generates methane — 80× more potent than CO₂.",
    icon: "🍂",
  },
  {
    tip: "Recycling 1 ton of paper saves 17 trees and 7,000 gallons of water.",
    icon: "📄",
  },
  {
    tip: "Americans use over 60 million plastic water bottles a day.",
    icon: "🧴",
  },
  {
    tip: 'The "Reduce, Reuse, Recycle" hierarchy prioritizes reducing waste first.',
    icon: "💡",
  },
  {
    tip: "A glass bottle made from recycled glass reduces air pollution by 20%.",
    icon: "🍾",
  },
];

// ═══════════════════════════════════════════════════════════════════
// 1. PARTICLE BACKGROUND
// ═══════════════════════════════════════════════════════════════════

class ParticleSystem {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.particles = [];
    this.symbols = ["♻️", "🌱", "🌍", "🍃", "💧", "⚡", "🌿"];
    this.maxParticles = 25;
    this.resize();
    window.addEventListener("resize", () => this.resize());
    this.init();
    this.animate();
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  init() {
    for (let i = 0; i < this.maxParticles; i++) {
      this.particles.push(this.createParticle());
    }
  }

  createParticle() {
    return {
      x: Math.random() * this.canvas.width,
      y: Math.random() * this.canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: -Math.random() * 0.4 - 0.1,
      symbol: this.symbols[Math.floor(Math.random() * this.symbols.length)],
      size: Math.random() * 16 + 10,
      opacity: Math.random() * 0.12 + 0.03,
      rotation: Math.random() * 360,
      rotationSpeed: (Math.random() - 0.5) * 0.5,
    };
  }

  animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    for (const p of this.particles) {
      p.x += p.vx;
      p.y += p.vy;
      p.rotation += p.rotationSpeed;

      if (p.y < -30) p.y = this.canvas.height + 30;
      if (p.x < -30) p.x = this.canvas.width + 30;
      if (p.x > this.canvas.width + 30) p.x = -30;

      this.ctx.save();
      this.ctx.translate(p.x, p.y);
      this.ctx.rotate((p.rotation * Math.PI) / 180);
      this.ctx.globalAlpha = p.opacity;
      this.ctx.font = `${p.size}px serif`;
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      this.ctx.fillText(p.symbol, 0, 0);
      this.ctx.restore();
    }

    requestAnimationFrame(() => this.animate());
  }
}

// ═══════════════════════════════════════════════════════════════════
// 2. IMAGE HANDLER
// ═══════════════════════════════════════════════════════════════════

class ImageHandler {
  constructor(app) {
    this.app = app;
    this.uploadZone = document.getElementById("upload-zone");
    this.fileInput = document.getElementById("file-input");
    this.btnUpload = document.getElementById("btn-upload");
    this.btnPredict = document.getElementById("btn-predict");
    this.btnReset = document.getElementById("btn-reset");
    this.previewContainer = document.getElementById("preview-container");
    this.previewImage = document.getElementById("preview-image");
    this.previewOverlay = document.getElementById("preview-overlay");
    this.currentFile = null;

    this.bindEvents();
  }

  bindEvents() {
    // Upload button
    this.btnUpload.addEventListener("click", (e) => {
      e.stopPropagation();
      this.fileInput.click();
    });

    // Click on zone (but not on buttons or preview)
    this.uploadZone.addEventListener("click", (e) => {
      if (
        e.target === this.uploadZone ||
        e.target.closest(".upload-icon") ||
        e.target.closest(".upload-title") ||
        e.target.closest(".upload-desc")
      ) {
        this.fileInput.click();
      }
    });

    // Keyboard accessibility
    this.uploadZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        this.fileInput.click();
      }
    });

    // File selected
    this.fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) this.processFile(file);
    });

    // Drag and drop
    this.uploadZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.uploadZone.classList.add("dragover");
    });

    this.uploadZone.addEventListener("dragleave", () => {
      this.uploadZone.classList.remove("dragover");
    });

    this.uploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      this.uploadZone.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        this.processFile(file);
      } else {
        this.app.showToast("⚠️ Please upload a valid image file.", "error");
      }
    });

    // Predict button
    this.btnPredict.addEventListener("click", (e) => {
      e.stopPropagation();
      if (this.currentFile) {
        this.app.classifyImage(this.currentFile);
      }
    });

    // Reset button
    this.btnReset.addEventListener("click", (e) => {
      e.stopPropagation();
      this.reset();
    });
  }

  processFile(file) {
    // Validate file type
    const validTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
      "image/bmp",
      "image/gif",
    ];
    if (!validTypes.includes(file.type)) {
      this.app.showToast("⚠️ Invalid file type. Use JPG, PNG, or WebP.", "error");
      return;
    }

    // Validate file size (16 MB)
    if (file.size > 16 * 1024 * 1024) {
      this.app.showToast("⚠️ File too large. Maximum size is 16 MB.", "error");
      return;
    }

    this.currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      this.showPreview(e.target.result);
    };
    reader.readAsDataURL(file);
  }

  showPreview(src) {
    this.previewImage.src = src;
    this.previewContainer.classList.add("active");
    this.btnPredict.disabled = false;
  }

  startScanning() {
    this.previewOverlay.classList.add("scanning");
    this.btnPredict.disabled = true;
    this.btnPredict.innerHTML =
      '<span class="scan-spinner" style="width:16px;height:16px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px"></span> Analyzing...';
  }

  stopScanning() {
    this.previewOverlay.classList.remove("scanning");
    this.btnPredict.disabled = false;
    this.btnPredict.innerHTML =
      '<span class="btn-icon">🔍</span> Classify Waste';
  }

  reset() {
    this.previewContainer.classList.remove("active");
    this.previewImage.src = "";
    this.currentFile = null;
    this.fileInput.value = "";
    this.btnPredict.disabled = false;
    this.btnPredict.innerHTML =
      '<span class="btn-icon">🔍</span> Classify Waste';
    this.stopScanning();
  }
}

// ═══════════════════════════════════════════════════════════════════
// 3. API CLIENT
// ═══════════════════════════════════════════════════════════════════

class APIClient {
  /**
   * Send an image to the Flask backend for classification.
   * @param {File} imageFile - The image file to classify
   * @returns {Promise<Object>} - The prediction response
   */
  async predict(imageFile) {
    const formData = new FormData();
    formData.append("image", imageFile);

    const response = await fetch(PREDICT_URL, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || `Server error (${response.status})`);
    }

    return data;
  }

  /**
   * Check if the backend server is healthy.
   * @returns {Promise<Object>} - Health status
   */
  async healthCheck() {
    try {
      const response = await fetch(HEALTH_URL, { method: "GET" });
      if (!response.ok) throw new Error("Health check failed");
      return await response.json();
    } catch (err) {
      return { status: "unreachable", model_loaded: false };
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// 4. UI CONTROLLER
// ═══════════════════════════════════════════════════════════════════

class UIController {
  constructor(app) {
    this.app = app;
  }

  showResults(apiResponse) {
    const section = document.getElementById("results-section");
    const prediction = apiResponse.prediction;
    const recommendation = apiResponse.recommendation;
    const predictedClass = prediction.class;
    const cat = CATEGORY_META[predictedClass] || CATEGORY_META["Non-Recyclable"];

    // Scroll to results
    section.classList.add("active");
    setTimeout(() => {
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);

    // Set accent color on card
    const card = document.getElementById("result-card");
    card.style.setProperty("--result-accent", cat.gradient);

    // Thumbnail
    const thumb = document.getElementById("result-thumb");
    thumb.src = document.getElementById("preview-image").src;

    // Item name (predicted class)
    document.getElementById("result-item-name").textContent = predictedClass;

    // Badge
    const badge = document.getElementById("result-badge");
    badge.textContent = `${cat.icon} ${predictedClass}`;
    badge.style.background = `${cat.color}22`;
    badge.style.color = cat.color;
    badge.style.border = `1px solid ${cat.color}44`;

    // Recommendation text
    const recEl = document.getElementById("result-recommendation");
    recEl.textContent = `${recommendation.icon} ${recommendation.recommendation}`;
    recEl.style.borderLeftColor = cat.color;

    // Confidence ring
    this.animateConfidence(prediction.confidence);

    // All predictions breakdown
    this.renderPredictionBars(prediction.all_predictions);

    // Recycling steps
    const stepsList = document.getElementById("recycle-steps");
    stepsList.innerHTML = "";
    for (const step of recommendation.details) {
      const li = document.createElement("li");
      li.textContent = step;
      stepsList.appendChild(li);
    }

    // Don'ts
    const dontsList = document.getElementById("recycle-donts");
    dontsList.innerHTML = "";
    for (const dont of recommendation.donts) {
      const li = document.createElement("li");
      li.textContent = dont;
      dontsList.appendChild(li);
    }

    // Factoid
    document.getElementById("factoid-text").textContent =
      recommendation.impact.fact;
  }

  renderPredictionBars(allPredictions) {
    const container = document.getElementById("prediction-bars");
    container.innerHTML = "";

    // Sort by confidence descending
    const sorted = Object.entries(allPredictions).sort(
      (a, b) => b[1].confidence - a[1].confidence
    );

    for (const [className, data] of sorted) {
      const cat =
        CATEGORY_META[className] || CATEGORY_META["Non-Recyclable"];
      const pct = (data.confidence * 100).toFixed(1);

      const item = document.createElement("div");
      item.className = "prediction-bar-item";
      item.innerHTML = `
        <span class="prediction-bar-label">${cat.icon} ${className}</span>
        <div class="prediction-bar-track">
          <div class="prediction-bar-fill" style="width:0%;background:${cat.color}" data-width="${pct}"></div>
        </div>
        <span class="prediction-bar-value">${pct}%</span>
      `;
      container.appendChild(item);
    }

    // Animate bars
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        container.querySelectorAll(".prediction-bar-fill").forEach((fill) => {
          fill.style.width = fill.dataset.width + "%";
        });
      });
    });
  }

  animateConfidence(confidence) {
    const ring = document.getElementById("confidence-ring-fill");
    const valueEl = document.getElementById("confidence-value");
    const circumference = 2 * Math.PI * 40; // r=40
    const offset = circumference - confidence * circumference;

    // Reset
    ring.style.transition = "none";
    ring.style.strokeDashoffset = circumference;
    valueEl.textContent = "0%";

    // Animate
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        ring.style.transition =
          "stroke-dashoffset 1.5s cubic-bezier(0.16, 1, 0.3, 1)";
        ring.style.strokeDashoffset = offset;

        // Counter animation
        const target = Math.round(confidence * 100);
        const duration = 1500;
        const startTime = performance.now();

        const tick = (now) => {
          const elapsed = now - startTime;
          const progress = Math.min(elapsed / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          const current = Math.round(eased * target);
          valueEl.textContent = `${current}%`;
          if (progress < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
    });
  }

  renderHistory() {
    const grid = document.getElementById("history-grid");
    const history = this.app.historyManager.history;

    grid.innerHTML = "";

    if (history.length === 0) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      empty.innerHTML =
        '<span class="history-empty-icon">📷</span> No items classified yet. Upload an image to get started!';
      grid.appendChild(empty);
      return;
    }

    for (const item of history) {
      const cat =
        CATEGORY_META[item.predictedClass] ||
        CATEGORY_META["Non-Recyclable"];

      const el = document.createElement("div");
      el.className = "history-item";
      el.style.setProperty("--item-accent", cat.color);

      const timeAgo = this.getTimeAgo(item.timestamp);

      el.innerHTML = `
        ${
          item.thumbnail
            ? `<img class="history-thumb" src="${item.thumbnail}" alt="${item.predictedClass}">`
            : ""
        }
        <div class="history-info">
          <div class="history-name">${item.predictedClass}</div>
          <span class="history-category" style="background:${cat.color}22;color:${cat.color}">${cat.icon} ${item.predictedClass}</span>
          <div class="history-confidence">${item.confidencePercent} confidence</div>
          <div class="history-time">${timeAgo}</div>
        </div>
      `;

      grid.appendChild(el);
    }
  }

  getTimeAgo(isoString) {
    const now = new Date();
    const then = new Date(isoString);
    const seconds = Math.floor((now - then) / 1000);

    if (seconds < 60) return "Just now";
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  }

  renderGuideCards() {
    const grid = document.getElementById("guide-grid");
    if (!grid) return;
    grid.innerHTML = "";

    for (const [className, cat] of Object.entries(CATEGORY_META)) {
      const card = document.createElement("div");
      card.className = "guide-card";
      card.style.setProperty("--card-gradient", cat.gradient);

      card.innerHTML = `
        <div class="guide-card-header">
          <span class="guide-card-icon">${cat.emoji}</span>
          <span class="guide-card-title">${className}</span>
        </div>
        <p class="guide-card-desc">${cat.description}</p>
        <span class="guide-card-bin">${cat.icon} Dispose appropriately</span>
      `;

      grid.appendChild(card);
    }
  }

  updateModelStatus(health) {
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("status-text");

    if (health.status === "unreachable") {
      dot.className = "status-dot error";
      text.textContent = "Server offline";
    } else if (health.model_loaded) {
      dot.className = "status-dot connected";
      text.textContent = "Model ready";
    } else {
      dot.className = "status-dot simulated";
      text.textContent = "Demo mode";
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// 5. HISTORY MANAGER
// ═══════════════════════════════════════════════════════════════════

class HistoryManager {
  constructor() {
    this.storageKey = "ecoscan_flask_history";
    this.history = this.load();
  }

  load() {
    try {
      const data = localStorage.getItem(this.storageKey);
      return data ? JSON.parse(data) : [];
    } catch {
      return [];
    }
  }

  save() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.history));
    } catch {
      // Storage full or unavailable
    }
  }

  add(entry) {
    this.history.unshift({
      id: Date.now(),
      predictedClass: entry.predictedClass,
      confidence: entry.confidence,
      confidencePercent: entry.confidencePercent,
      recommendation: entry.recommendation,
      thumbnail: entry.thumbnail,
      modelLoaded: entry.modelLoaded,
      timestamp: new Date().toISOString(),
    });
    // Keep last 30
    if (this.history.length > 30) this.history = this.history.slice(0, 30);
    this.save();
  }

  clear() {
    this.history = [];
    this.save();
  }
}

// ═══════════════════════════════════════════════════════════════════
// 6. ECO TICKER
// ═══════════════════════════════════════════════════════════════════

class EcoTicker {
  constructor() {
    this.textEl = document.getElementById("eco-ticker-text");
    this.iconEl = document.getElementById("eco-ticker-icon");
    if (!this.textEl || !this.iconEl) return;
    this.currentIndex = Math.floor(Math.random() * ECO_TIPS.length);
    this.show();
    setInterval(() => this.next(), 8000);
  }

  show() {
    const tip = ECO_TIPS[this.currentIndex];
    this.textEl.textContent = tip.tip;
    this.iconEl.textContent = tip.icon;
  }

  next() {
    this.currentIndex = (this.currentIndex + 1) % ECO_TIPS.length;
    const content = document.getElementById("eco-ticker-content");
    if (content) {
      content.style.animation = "none";
      content.offsetHeight; // force reflow
      content.style.animation = "";
    }
    this.show();
  }
}

// ═══════════════════════════════════════════════════════════════════
// 7. MAIN APP
// ═══════════════════════════════════════════════════════════════════

class EcoScanApp {
  constructor() {
    this.apiClient = new APIClient();
    this.historyManager = new HistoryManager();
    this.uiController = new UIController(this);
    this.imageHandler = new ImageHandler(this);
    this.particleSystem = new ParticleSystem("particle-canvas");
    this.ecoTicker = new EcoTicker();

    this.init();
  }

  async init() {
    // Render static sections
    this.uiController.renderGuideCards();
    this.uiController.renderHistory();

    // Setup clear history
    const clearBtn = document.getElementById("btn-clear-history");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        this.historyManager.clear();
        this.uiController.renderHistory();
        this.showToast("🗑️ History cleared", "success");
      });
    }

    // Classify another button
    const classifyAnotherBtn = document.getElementById("btn-classify-another");
    if (classifyAnotherBtn) {
      classifyAnotherBtn.addEventListener("click", () => {
        this.imageHandler.reset();
        document.getElementById("results-section").classList.remove("active");
        document
          .getElementById("upload-section")
          .scrollIntoView({ behavior: "smooth", block: "center" });
      });
    }

    // Check server health
    const health = await this.apiClient.healthCheck();
    this.uiController.updateModelStatus(health);
  }

  async classifyImage(imageFile) {
    this.imageHandler.startScanning();

    try {
      // Small delay for UX (scanning animation)
      await new Promise((r) => setTimeout(r, 800));

      // Call Flask backend
      const response = await this.apiClient.predict(imageFile);

      // Stop scanning
      this.imageHandler.stopScanning();

      // Create thumbnail for history
      const thumbnail = await this.createThumbnail(
        this.imageHandler.previewImage.src
      );

      // Add to history
      this.historyManager.add({
        predictedClass: response.prediction.class,
        confidence: response.prediction.confidence,
        confidencePercent: response.prediction.confidence_percent,
        recommendation: response.recommendation.recommendation,
        modelLoaded: response.model_loaded,
        thumbnail: thumbnail,
      });

      // Update UI
      this.uiController.showResults(response);
      this.uiController.renderHistory();

      const cat =
        CATEGORY_META[response.prediction.class] ||
        CATEGORY_META["Non-Recyclable"];
      this.showToast(
        `${cat.icon} Classified as ${response.prediction.class}! (${response.prediction.confidence_percent})`,
        "success"
      );
    } catch (err) {
      console.error("Classification error:", err);
      this.imageHandler.stopScanning();
      this.showToast(`❌ ${err.message || "Classification failed"}`, "error");
    }
  }

  createThumbnail(imageSrc) {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 80;
        canvas.height = 80;
        const ctx = canvas.getContext("2d");
        const size = Math.min(img.naturalWidth, img.naturalHeight);
        const sx = (img.naturalWidth - size) / 2;
        const sy = (img.naturalHeight - size) / 2;
        ctx.drawImage(img, sx, sy, size, size, 0, 0, 80, 80);
        resolve(canvas.toDataURL("image/jpeg", 0.7));
      };
      img.onerror = () => resolve(null);
      img.src = imageSrc;
    });
  }

  showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    const icon = document.getElementById("toast-icon");
    const text = document.getElementById("toast-text");

    if (!toast || !icon || !text) return;

    icon.textContent = type === "success" ? "✅" : "⚠️";
    text.textContent = message;
    toast.classList.add("active");

    setTimeout(() => {
      toast.classList.remove("active");
    }, 4000);
  }
}

// ─── Bootstrap ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  window.ecoscanApp = new EcoScanApp();
});
