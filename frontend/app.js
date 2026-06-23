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
// 7. ECOBOT WIDGET CONTROLLER
// ═══════════════════════════════════════════════════════════════════

const GREETING_PATTERNS = /\b(hi|hello|hey|greetings|howdy|good\s*(morning|afternoon|evening))\b/i;
const FAREWELL_PATTERNS = /\b(bye|goodbye|see\s*you|thanks|thank\s*you|cheers|exit|quit)\b/i;

const WASTE_KNOWLEDGE = {
  plastic: {
    patterns: [/\bplastic\b/i, /\bPET\b/i, /\bHDPE\b/i, /\bpolythene\b/i, /\bplastic\s*bottle\b/i, /\bplastic\s*bag\b/i, /\bplastic\s*container\b/i, /\bstyrofoam\b/i, /\bpolystyrene\b/i],
    responses: [
      "♻️ **Plastic Recycling Guide**\n\n• Rinse containers before recycling\n• Check the resin code (1-7) on the bottom\n• PET (#1) and HDPE (#2) are most widely recyclable\n• **Never** put plastic bags in curbside recycling — they jam machinery\n• Plastic bottles can be recycled into polyester fiber for clothing\n\n💡 **Tip:** Switch to reusable bottles and bags to reduce plastic waste by up to 90%!",
      "🌍 **Plastic Environmental Impact**\n\n• Decomposition time: **450+ years**\n• Only ~9% of plastic ever produced has been recycled\n• 8 million tons of plastic enter oceans annually\n• Microplastics have been found in drinking water, food, and human blood\n\n♻️ **Action:** Reduce plastic usage first, reuse second, recycle last."
    ]
  },
  paper: {
    patterns: [/\bpaper\b/i, /\bcardboard\b/i, /\bnewspaper\b/i, /\bmagazine\b/i, /\bcarton\b/i, /\benvelope\b/i],
    responses: [
      "📄 **Paper Recycling Guide**\n\n• Keep paper clean and dry — grease/food ruins recyclability\n• Flatten cardboard boxes before recycling\n• Paper can be recycled up to **7 times** before fibers are too short\n• Remove plastic windows from envelopes\n• Pizza boxes with grease stains should be composted, not recycled\n\n💡 **Tip:** Recycling 1 ton of paper saves 17 trees, 7,000 gallons of water!"
    ]
  },
  glass: {
    patterns: [/\bglass\b/i, /\bbottle\b.*\bglass\b/i, /\bglass\s*jar\b/i, /\bglass\s*container\b/i],
    responses: [
      "🫙 **Glass Recycling Guide**\n\n• Glass is **100% recyclable** with zero quality loss — infinitely!\n• Rinse jars and bottles, remove metal caps\n• Separate by color if required (clear, green, brown)\n• Do NOT include ceramics, mirrors, or Pyrex (different melting points)\n• Decomposition time if landfilled: **1 million years**\n\n💡 **Tip:** Reuse glass jars for food storage before recycling."
    ]
  },
  metal: {
    patterns: [/\bmetal\b/i, /\baluminum\b/i, /\baluminium\b/i, /\bsteel\b/i, /\btin\b/i, /\bcan\b/i, /\bcopper\b/i, /\biron\b/i],
    responses: [
      "🔩 **Metal Recycling Guide**\n\n• Rinse food cans and crush to save bin space\n• Aluminum is the most widely recycled — saves **95% energy**\n• Use a magnet to sort: sticks = steel, doesn't stick = aluminum\n• Include clean aluminum foil and trays\n• Scrap metal can be sold to dedicated recyclers\n\n💡 **Tip:** Recycling one aluminum can saves enough energy to run a TV for 3 hours!"
    ]
  },
  organic: {
    patterns: [/\borganic\b/i, /\bfood\s*(waste|scrap)\b/i, /\bcompost\b/i, /\bbiodegradable\b/i, /\bfruit\b.*\bpeel\b/i, /\bvegetable\b/i, /\bleaves\b/i, /\bgrass\b/i, /\bgarden\s*waste\b/i],
    responses: [
      "🌱 **Organic Waste & Composting Guide**\n\n• Food scraps, peels, coffee grounds, eggshells → compost!\n• Yard waste (leaves, grass, branches) → compost or green bin\n• Home composting produces free, nutrient-rich fertilizer\n• **Avoid** composting meat, dairy, and oily foods at home\n• Decomposition time: **2–4 weeks** (fastest of all categories)\n\n💡 **Tip:** Composting diverts organics from landfills and reduces methane emissions by 50%!"
    ]
  },
  ewaste: {
    patterns: [/\be[\-\s]*waste\b/i, /\belectronic\b/i, /\bbatter(y|ies)\b/i, /\bphone\b/i, /\bcomputer\b/i, /\blaptop\b/i, /\btablet\b/i, /\bcharger\b/i, /\bcable\b/i, /\blight\s*bulb\b/i, /\bLED\b/i, /\bcircuit\b/i, /\bprinter\b/i, /\bmonitor\b/i],
    responses: [
      "⚡ **E-Waste Disposal Guide**\n\n• **NEVER** place e-waste in regular trash — contains toxic heavy metals!\n• Locate certified e-waste recyclers (e-Stewards, R2 certified)\n• Remove batteries separately and recycle at drop-off points\n• Wipe personal data before recycling devices\n• Check manufacturer take-back programs (Apple, Dell, HP, Samsung)\n\n⚠️ **Warning:** E-waste contains lead, mercury, and cadmium that cause severe soil and groundwater contamination."
    ]
  }
};

const GENERAL_KNOWLEDGE = {
  recycling_tips: {
    patterns: [/\brecycl(e|ing)\s*tip\b/i, /\bhow\s*to\s*recycle\b/i, /\brecycling\s*guide\b/i, /\bwhat\s*can\s*(i|we)\s*recycle\b/i],
    responses: [
      "♻️ **Top 10 Recycling Tips**\n\n1. **Rinse** containers before recycling\n2. **Flatten** cardboard and boxes\n3. **Check** local guidelines — rules vary by area\n4. **Remove** caps and lids when required\n5. **Keep it clean** — contamination ruins entire batches\n6. **No plastic bags** in curbside bins\n7. **Separate** glass by color if required\n8. **Compost** food waste instead of trashing it\n9. **E-waste** goes to specialized facilities only\n10. **Reduce & Reuse** before recycling!"
    ]
  },
  environment: {
    patterns: [/\benvironment\b/i, /\bclimate\b/i, /\bglobal\s*warming\b/i, /\bcarbon\s*footprint\b/i, /\bsustain\b/i, /\bgreen\b/i, /\beco\s*friendly\b/i, /\bplanet\b/i],
    responses: [
      "🌍 **Environmental Awareness**\n\n• Waste management accounts for ~5% of global greenhouse gas emissions\n• Landfills produce methane — 25x more potent than CO₂\n• Recycling aluminum saves 95% of energy vs. raw production\n• The Great Pacific Garbage Patch is now 3x the size of France\n• By 2050, oceans may contain more plastic than fish by weight\n\n🌱 **What YOU can do:**\n• Reduce consumption\n• Choose reusable products\n• Sort waste properly\n• Support circular economy initiatives"
    ]
  },
  reduce: {
    patterns: [/\breduce\b/i, /\bminimize\b/i, /\bless\s*waste\b/i, /\bzero\s*waste\b/i, /\bwaste\s*reduction\b/i],
    responses: [
      "📉 **Waste Reduction Strategies**\n\n1. **Refuse** what you don't need\n2. **Reduce** what you use\n3. **Reuse** before discarding\n4. **Repurpose** items creatively\n5. **Recycle** what you can't reuse\n6. **Rot** (compost) organic waste\n\n💡 The most sustainable waste is the waste never created!"
    ]
  },
  decomposition: {
    patterns: [/\bdecompos(e|ition)\b/i, /\bhow\s*long\b.*\b(break|degrade|last)\b/i, /\bbiodegrad\b/i],
    responses: [
      "⏳ **Decomposition Timeline**\n\n| Material | Time |\n|---|---|\n| Organic waste | 2–4 weeks |\n| Paper | 2–6 weeks |\n| Cotton cloth | 1–5 months |\n| Tin can | 50 years |\n| Aluminum can | 200 years |\n| Plastic bottle | 450 years |\n| Glass bottle | 1 million years |\n| Styrofoam | Never |\n| E-waste | Never (toxic) |"
    ]
  }
};

const FALLBACK_RESPONSES = [
  "🤔 I'm not sure about that specific topic. I can help with:\n\n• **Plastic, Paper, Glass, Metal, Organic waste, E-waste** disposal\n• **Recycling tips** and guides\n• **Environmental impact** information\n• **Decomposition timelines**\n• **Waste reduction** strategies\n\nTry asking something like: *'How do I recycle plastic bottles?'*",
  "I specialize in waste management and recycling. Try asking me:\n\n• *'What should I do with e-waste?'*\n• *'How long does plastic take to decompose?'*\n• *'Give me recycling tips'*\n• *'Tell me about environmental impact'*"
];

const GREETING_RESPONSES = [
  "👋 Hello! I'm your AI Waste Management Assistant. Ask me anything about recycling, waste disposal, or environmental impact!",
  "Hi there! ♻️ I'm here to help with waste sorting, recycling guidance, and eco-tips. What would you like to know?",
  "Hey! 🌍 Ready to help you make greener choices. What's on your mind?"
];

const FAREWELL_RESPONSES = [
  "Thanks for caring about the environment! 🌱 Every small action counts. Goodbye!",
  "Bye! Remember: Reduce → Reuse → Recycle ♻️",
  "See you! Keep sorting your waste properly — the planet thanks you! 🌍"
];

class EcoBotController {
  constructor(app, isClientOnly = false) {
    this.app = app;
    this.isClientOnly = isClientOnly;
    
    this.launcher = document.getElementById("ecobot-launcher");
    this.popup = document.getElementById("ecobot-popup");
    this.closeBtn = document.getElementById("ecobot-close");
    this.form = document.getElementById("ecobot-form");
    this.input = document.getElementById("ecobot-input");
    this.messagesContainer = document.getElementById("ecobot-messages");
    this.statusText = document.getElementById("ecobot-status-text");
    this.statusDot = document.querySelector(".ecobot-status-dot");
    
    if (!this.launcher || !this.popup) return;
    
    this.isOpen = false;
    this.hasSentWelcome = false;
    
    this.bindEvents();
    this.updateStatus();
  }
  
  bindEvents() {
    this.launcher.addEventListener("click", () => this.toggle());
    this.closeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.close();
    });
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.sendMessage();
    });
  }
  
  updateStatus() {
    if (this.isClientOnly) {
      if (this.statusText) this.statusText.textContent = "Offline Mode";
      if (this.statusDot) {
        this.statusDot.className = "ecobot-status-dot offline";
      }
    } else {
      if (this.statusText) this.statusText.textContent = "Online";
      if (this.statusDot) {
        this.statusDot.className = "ecobot-status-dot online";
      }
    }
  }
  
  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }
  
  open() {
    this.isOpen = true;
    this.popup.classList.add("active");
    this.popup.setAttribute("aria-hidden", "false");
    
    if (!this.hasSentWelcome) {
      this.sendWelcomeMessage();
      this.hasSentWelcome = true;
    }
    
    // Autofocus input
    setTimeout(() => {
      if (this.input) this.input.focus();
    }, 150);
  }
  
  close() {
    this.isOpen = false;
    this.popup.classList.remove("active");
    this.popup.setAttribute("aria-hidden", "true");
  }
  
  sendWelcomeMessage() {
    const welcomeText = "👋 Hello! I'm EcoBot. Ask me anything about waste classification, recycling recommendations, sustainability, or waste management.";
    this.appendMessage("welcome-caption", welcomeText);
  }
  
  appendMessage(sender, text) {
    const msgEl = document.createElement("div");
    msgEl.className = `ecobot-msg ${sender}`;
    
    if (sender === "welcome-caption") {
      msgEl.textContent = text;
    } else {
      const formatted = this.formatMarkdown(text);
      msgEl.innerHTML = formatted;
      
      const timeEl = document.createElement("div");
      timeEl.className = "ecobot-msg-time";
      const now = new Date();
      timeEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      msgEl.appendChild(timeEl);
    }
    
    this.messagesContainer.appendChild(msgEl);
    this.scrollToBottom();
  }
  
  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }
  
  showTypingIndicator() {
    const indicator = document.createElement("div");
    indicator.className = "ecobot-typing";
    indicator.id = "ecobot-typing-indicator";
    indicator.innerHTML = `
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    `;
    this.messagesContainer.appendChild(indicator);
    this.scrollToBottom();
  }
  
  hideTypingIndicator() {
    const indicator = document.getElementById("ecobot-typing-indicator");
    if (indicator) {
      indicator.remove();
    }
  }
  
  async sendMessage() {
    const text = this.input.value.trim();
    if (!text) return;
    
    this.input.value = "";
    this.appendMessage("user", text);
    this.showTypingIndicator();
    
    // UX delay: typing simulation
    await new Promise((r) => setTimeout(r, 600));
    
    if (this.isClientOnly) {
      const reply = this.getLocalResponse(text);
      this.hideTypingIndicator();
      this.appendMessage("bot", reply);
    } else {
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ message: text })
        });
        
        if (response.ok) {
          const data = await response.json();
          this.hideTypingIndicator();
          if (data.success) {
            this.appendMessage("bot", data.response);
          } else {
            this.appendMessage("bot", "⚠️ Sorry, I encountered an issue processing that. " + this.getLocalResponse(text));
          }
        } else {
          throw new Error("HTTP error");
        }
      } catch (err) {
        console.warn("Backend chat failed, falling back to local client-side matching:", err);
        const reply = this.getLocalResponse(text);
        this.hideTypingIndicator();
        this.appendMessage("bot", reply);
      }
    }
  }
  
  formatMarkdown(text) {
    let html = text;
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Lists
    const lines = html.split('\n');
    let inList = false;
    let listType = null; // 'ul' or 'ol'
    let result = [];
    
    for (let line of lines) {
      let trimmed = line.trim();
      if (trimmed.startsWith('• ') || trimmed.startsWith('- ')) {
        if (!inList || listType !== 'ul') {
          if (inList) result.push(listType === 'ol' ? '</ol>' : '</ul>');
          result.push('<ul>');
          inList = true;
          listType = 'ul';
        }
        result.push(`<li>${trimmed.substring(2)}</li>`);
      } else if (/^\d+\.\s/.test(trimmed)) {
        if (!inList || listType !== 'ol') {
          if (inList) result.push(listType === 'ol' ? '</ol>' : '</ul>');
          result.push('<ol>');
          inList = true;
          listType = 'ol';
        }
        const textOnly = trimmed.replace(/^\d+\.\s/, '');
        result.push(`<li>${textOnly}</li>`);
      } else {
        if (inList) {
          result.push(listType === 'ol' ? '</ol>' : '</ul>');
          inList = false;
          listType = null;
        }
        if (trimmed.length > 0) {
          result.push(`<p>${line}</p>`);
        }
      }
    }
    if (inList) {
      result.push(listType === 'ol' ? '</ol>' : '</ul>');
    }
    
    return result.join('');
  }
  
  getLocalResponse(message) {
    const msg = message.toLowerCase().trim();
    
    if (GREETING_PATTERNS.test(msg)) {
      return GREETING_RESPONSES[Math.floor(Math.random() * GREETING_RESPONSES.length)];
    }
    
    if (FAREWELL_PATTERNS.test(msg)) {
      return FAREWELL_RESPONSES[Math.floor(Math.random() * FAREWELL_RESPONSES.length)];
    }
    
    for (const [key, data] of Object.entries(WASTE_KNOWLEDGE)) {
      for (const pattern of data.patterns) {
        if (pattern.test(msg)) {
          return data.responses[Math.floor(Math.random() * data.responses.length)];
        }
      }
    }
    
    for (const [key, data] of Object.entries(GENERAL_KNOWLEDGE)) {
      for (const pattern of data.patterns) {
        if (pattern.test(msg)) {
          return data.responses[Math.floor(Math.random() * data.responses.length)];
        }
      }
    }
    
    return FALLBACK_RESPONSES[Math.floor(Math.random() * FALLBACK_RESPONSES.length)];
  }
}

// ═══════════════════════════════════════════════════════════════════
// 8. MAIN APP
// ═══════════════════════════════════════════════════════════════════

class EcoScanApp {
  constructor() {
    this.apiClient = new APIClient();
    this.historyManager = new HistoryManager();
    this.uiController = new UIController(this);
    this.imageHandler = new ImageHandler(this);
    this.particleSystem = new ParticleSystem("particle-canvas");
    this.ecoTicker = new EcoTicker();
    this.ecoBot = new EcoBotController(this, false);

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
