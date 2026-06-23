/**
 * app.js — Main Application Logic
 * AI Waste Classification & Recycling Recommendation System
 * 
 * Modules:
 *  1. Particle Background
 *  2. Image Handler (drag-drop, upload, camera)
 *  3. AI Classification Engine (TensorFlow.js + MobileNet)
 *  4. Waste Category Mapper
 *  5. Recycling Recommendation Engine
 *  6. History & Stats Manager
 *  7. UI Controller (DOM, animations, charts)
 *  8. Eco Ticker
 */

const {
  CATEGORIES,
  CLASS_TO_CATEGORY,
  RECYCLING_INSTRUCTIONS,
  ENVIRONMENTAL_IMPACT,
  ECO_TIPS,
  DECOMPOSITION_TIMELINE,
} = window;


// ═══════════════════════════════════════════════════════════════
// 1. PARTICLE BACKGROUND
// ═══════════════════════════════════════════════════════════════
class ParticleSystem {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.particles = [];
    this.symbols = ['♻️', '🌱', '🌍', '🍃', '💧', '⚡', '🌿'];
    this.maxParticles = 30;
    this.resize();
    window.addEventListener('resize', () => this.resize());
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
      opacity: Math.random() * 0.15 + 0.03,
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

      // Wrap around
      if (p.y < -30) p.y = this.canvas.height + 30;
      if (p.x < -30) p.x = this.canvas.width + 30;
      if (p.x > this.canvas.width + 30) p.x = -30;

      this.ctx.save();
      this.ctx.translate(p.x, p.y);
      this.ctx.rotate((p.rotation * Math.PI) / 180);
      this.ctx.globalAlpha = p.opacity;
      this.ctx.font = `${p.size}px serif`;
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(p.symbol, 0, 0);
      this.ctx.restore();
    }

    requestAnimationFrame(() => this.animate());
  }
}

// ═══════════════════════════════════════════════════════════════
// 2. IMAGE HANDLER
// ═══════════════════════════════════════════════════════════════
class ImageHandler {
  constructor(app) {
    this.app = app;
    this.uploadZone = document.getElementById('upload-zone');
    this.fileInput = document.getElementById('file-input');
    this.btnUpload = document.getElementById('btn-upload');
    this.btnCamera = document.getElementById('btn-camera');
    this.previewContainer = document.getElementById('preview-container');
    this.previewImage = document.getElementById('preview-image');
    this.previewOverlay = document.getElementById('preview-overlay');

    this.bindEvents();
  }

  bindEvents() {
    // Upload button
    this.btnUpload.addEventListener('click', (e) => {
      e.stopPropagation();
      this.fileInput.click();
    });

    // Click on zone
    this.uploadZone.addEventListener('click', (e) => {
      if (e.target === this.uploadZone || e.target.closest('.upload-icon') || e.target.closest('.upload-title') || e.target.closest('.upload-desc')) {
        this.fileInput.click();
      }
    });

    // Keyboard
    this.uploadZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        this.fileInput.click();
      }
    });

    // File selected
    this.fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) this.processFile(file);
    });

    // Drag and drop
    this.uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.uploadZone.classList.add('dragover');
    });

    this.uploadZone.addEventListener('dragleave', () => {
      this.uploadZone.classList.remove('dragover');
    });

    this.uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadZone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        this.processFile(file);
      }
    });

    // Camera button
    this.btnCamera.addEventListener('click', (e) => {
      e.stopPropagation();
      this.app.cameraHandler.open();
    });
  }

  processFile(file) {
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        await this.showPreview(e.target.result);
        this.app.classifyImage(this.previewImage);
      } catch (err) {
        this.stopScanning();
        this.app.showToast('⚠️ Could not load that image. Please try another file.', 'error');
      }
    };
    reader.readAsDataURL(file);
  }

  showPreview(src) {
    this.previewContainer.classList.add('active');
    this.startScanning();

    return new Promise((resolve, reject) => {
      const cleanup = () => {
        this.previewImage.onload = null;
        this.previewImage.onerror = null;
      };

      this.previewImage.onload = () => {
        cleanup();
        resolve();
      };
      this.previewImage.onerror = () => {
        cleanup();
        reject(new Error('Preview image failed to load'));
      };

      this.previewImage.src = src;

      if (this.previewImage.complete && this.previewImage.naturalWidth > 0) {
        cleanup();
        resolve();
      }
    });
  }

  startScanning() {
    this.previewOverlay.classList.add('scanning');
  }

  stopScanning() {
    this.previewOverlay.classList.remove('scanning');
  }
}

// ═══════════════════════════════════════════════════════════════
// 3. CAMERA HANDLER
// ═══════════════════════════════════════════════════════════════
class CameraHandler {
  constructor(app) {
    this.app = app;
    this.modal = document.getElementById('camera-modal');
    this.video = document.getElementById('camera-video');
    this.canvas = document.getElementById('camera-canvas');
    this.btnCapture = document.getElementById('btn-capture');
    this.btnClose = document.getElementById('btn-close-camera');
    this.stream = null;

    this.btnCapture.addEventListener('click', () => this.capture());
    this.btnClose.addEventListener('click', () => this.close());
  }

  async open() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      this.video.srcObject = this.stream;
      this.modal.classList.add('active');
    } catch (err) {
      this.app.showToast('📷 Camera access denied. Please allow camera permissions.', 'error');
    }
  }

  async capture() {
    const ctx = this.canvas.getContext('2d');
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
    ctx.drawImage(this.video, 0, 0);

    const dataUrl = this.canvas.toDataURL('image/jpeg', 0.9);
    this.close();

    try {
      await this.app.imageHandler.showPreview(dataUrl);
      this.app.classifyImage(this.app.imageHandler.previewImage);
    } catch (err) {
      this.app.imageHandler.stopScanning();
      this.app.showToast('⚠️ Could not load the captured image. Please try again.', 'error');
    }
  }

  close() {
    this.modal.classList.remove('active');
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// 4. AI CLASSIFICATION ENGINE
// ═══════════════════════════════════════════════════════════════
class ClassificationEngine {
  constructor() {
    this.model = null;
    this.loading = false;
    this.loaded = false;
  }

  async loadModel() {
    if (this.loaded) return;
    if (this.loading) {
      // Wait for existing load
      while (this.loading) {
        await new Promise((r) => setTimeout(r, 100));
      }
      return;
    }
    this.loading = true;
    try {
      // mobilenet is loaded globally via CDN script tag
      this.model = await mobilenet.load({ version: 2, alpha: 1.0 });
      this.loaded = true;
    } catch (err) {
      console.error('Failed to load MobileNet:', err);
      throw err;
    } finally {
      this.loading = false;
    }
  }

  async classify(imageElement) {
    await this.loadModel();
    const predictions = await this.model.classify(imageElement, 10);
    return predictions;
  }
}

// ═══════════════════════════════════════════════════════════════
// 5. WASTE MAPPER
// ═══════════════════════════════════════════════════════════════
class WasteMapper {
  /**
   * Maps MobileNet predictions to waste categories.
   * Returns { category, itemName, confidence, rawPredictions }
   */
  map(predictions) {
    if (!predictions || predictions.length === 0) {
      return {
        category: 'general',
        itemName: 'Unknown Item',
        confidence: 0,
        rawPredictions: [],
      };
    }

    // Try each prediction in order of confidence
    for (const pred of predictions) {
      const className = pred.className.toLowerCase();

      // Check exact and partial matches
      for (const [key, category] of Object.entries(CLASS_TO_CATEGORY)) {
        if (className.includes(key.toLowerCase())) {
          return {
            category,
            itemName: this.formatName(pred.className),
            confidence: pred.probability,
            rawPredictions: predictions,
          };
        }
      }
    }

    // No match found — use top prediction with "general" category
    return {
      category: 'general',
      itemName: this.formatName(predictions[0].className),
      confidence: predictions[0].probability,
      rawPredictions: predictions,
    };
  }

  formatName(name) {
    // MobileNet names often have comma-separated variants; take first
    const primary = name.split(',')[0].trim();
    return primary
      .split(' ')
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ');
  }
}

// ═══════════════════════════════════════════════════════════════
// 6. HISTORY & STATS MANAGER
// ═══════════════════════════════════════════════════════════════
class HistoryManager {
  constructor() {
    this.storageKey = 'ecoscan_history';
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
      itemName: entry.itemName,
      category: entry.category,
      confidence: entry.confidence,
      thumbnail: entry.thumbnail,
      timestamp: new Date().toISOString(),
    });
    // Keep last 50
    if (this.history.length > 50) this.history = this.history.slice(0, 50);
    this.save();
  }

  clear() {
    this.history = [];
    this.save();
  }

  getStats() {
    const stats = {
      total: this.history.length,
      co2Saved: 0,
      energySaved: 0,
      waterSaved: 0,
      categoryCounts: {},
    };

    for (const item of this.history) {
      const impact = ENVIRONMENTAL_IMPACT[item.category];
      if (impact) {
        stats.co2Saved += impact.co2SavedPerKg * 0.5; // assume ~0.5kg avg item
        stats.energySaved += impact.energySavedPerKg * 0.5;
        stats.waterSaved += impact.waterSavedPerKg * 0.5;
      }
      stats.categoryCounts[item.category] = (stats.categoryCounts[item.category] || 0) + 1;
    }

    return stats;
  }
}

// ═══════════════════════════════════════════════════════════════
// 7. UI CONTROLLER
// ═══════════════════════════════════════════════════════════════
class UIController {
  constructor(app) {
    this.app = app;
  }

  showResults(result) {
    const section = document.getElementById('results-section');
    const cat = CATEGORIES[result.category];
    const instructions = RECYCLING_INSTRUCTIONS[result.category];
    const impact = ENVIRONMENTAL_IMPACT[result.category];

    // Scroll to results
    section.classList.add('active');
    setTimeout(() => {
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Set accent color on card
    const card = document.getElementById('result-card');
    card.style.setProperty('--result-accent', cat.gradient);

    // Thumbnail
    const thumb = document.getElementById('result-thumb');
    thumb.src = document.getElementById('preview-image').src;

    // Item name
    document.getElementById('result-item-name').textContent = result.itemName;

    // Badge
    const badge = document.getElementById('result-badge');
    badge.textContent = `${cat.icon} ${cat.name}`;
    badge.style.background = `${cat.color}22`;
    badge.style.color = cat.color;
    badge.style.border = `1px solid ${cat.color}44`;

    // Confidence ring
    this.animateConfidence(result.confidence);

    // Recycling steps
    const stepsList = document.getElementById('recycle-steps');
    stepsList.innerHTML = '';
    for (const step of instructions.steps) {
      const li = document.createElement('li');
      li.textContent = step;
      stepsList.appendChild(li);
    }

    // Don'ts
    const dontsList = document.getElementById('recycle-donts');
    dontsList.innerHTML = '';
    for (const dont of instructions.doNot) {
      const li = document.createElement('li');
      li.textContent = dont;
      dontsList.appendChild(li);
    }

    // Impact grid
    const impactGrid = document.getElementById('impact-grid');
    impactGrid.innerHTML = `
      <div class="impact-stat">
        <div class="impact-stat-icon">🌿</div>
        <span class="impact-stat-value">${impact.co2SavedPerKg} kg</span>
        <span class="impact-stat-label">CO₂ Saved/kg</span>
      </div>
      <div class="impact-stat">
        <div class="impact-stat-icon">⚡</div>
        <span class="impact-stat-value">${impact.energySavedPerKg} kWh</span>
        <span class="impact-stat-label">Energy Saved/kg</span>
      </div>
      <div class="impact-stat">
        <div class="impact-stat-icon">💧</div>
        <span class="impact-stat-value">${impact.waterSavedPerKg} L</span>
        <span class="impact-stat-label">Water Saved/kg</span>
      </div>
      <div class="impact-stat">
        <div class="impact-stat-icon">⏳</div>
        <span class="impact-stat-value" style="font-size:0.9rem">${impact.decompositionTime}</span>
        <span class="impact-stat-label">Decomposition</span>
      </div>
    `;

    // Tips
    const tipsList = document.getElementById('recycle-tips');
    tipsList.innerHTML = '';
    for (const tip of instructions.tips) {
      const li = document.createElement('li');
      li.textContent = tip;
      tipsList.appendChild(li);
    }

    // Factoid
    document.getElementById('factoid-text').textContent = impact.factoid;
  }

  animateConfidence(confidence) {
    const ring = document.getElementById('confidence-ring-fill');
    const valueEl = document.getElementById('confidence-value');
    const circumference = 2 * Math.PI * 40; // r=40
    const offset = circumference - (confidence * circumference);

    // Reset
    ring.style.transition = 'none';
    ring.style.strokeDashoffset = circumference;
    valueEl.textContent = '0%';

    // Animate
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        ring.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.16, 1, 0.3, 1)';
        ring.style.strokeDashoffset = offset;

        // Counter animation
        let start = 0;
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

  renderHistory(filter = 'all') {
    const grid = document.getElementById('history-grid');
    const emptyMsg = document.getElementById('history-empty');
    const items = filter === 'all'
      ? this.app.historyManager.history
      : this.app.historyManager.history.filter((i) => i.category === filter);

    grid.innerHTML = '';

    if (items.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'history-empty';
      empty.innerHTML = filter === 'all'
        ? '<span class="history-empty-icon">📷</span> No items classified yet. Upload an image to get started!'
        : `<span class="history-empty-icon">🔍</span> No ${CATEGORIES[filter]?.name || ''} items found.`;
      grid.appendChild(empty);
      return;
    }

    for (const item of items) {
      const cat = CATEGORIES[item.category] || CATEGORIES.general;
      const el = document.createElement('div');
      el.className = 'history-item';
      el.style.setProperty('--item-accent', cat.color);

      const timeAgo = this.getTimeAgo(item.timestamp);

      el.innerHTML = `
        ${item.thumbnail ? `<img class="history-thumb" src="${item.thumbnail}" alt="${item.itemName}">` : ''}
        <div class="history-info">
          <div class="history-name">${item.itemName}</div>
          <span class="history-category" style="background:${cat.color}22;color:${cat.color}">${cat.icon} ${cat.name}</span>
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

    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  }

  updateDashboard() {
    const stats = this.app.historyManager.getStats();

    document.getElementById('stat-total-scans').textContent = stats.total;
    document.getElementById('stat-co2-saved').textContent = `${stats.co2Saved.toFixed(1)} kg`;
    document.getElementById('dash-total').textContent = stats.total;
    document.getElementById('dash-co2').textContent = `${stats.co2Saved.toFixed(1)} kg`;
    document.getElementById('dash-energy').textContent = `${stats.energySaved.toFixed(1)} kWh`;
    document.getElementById('dash-water').textContent = `${stats.waterSaved.toFixed(1)} L`;

    this.renderDonutChart(stats.categoryCounts);
  }

  renderDonutChart(categoryCounts) {
    const canvas = document.getElementById('donut-chart');
    const ctx = canvas.getContext('2d');
    const legendContainer = document.getElementById('chart-legend');

    // DPR for sharp rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 220 * dpr;
    canvas.height = 220 * dpr;
    ctx.scale(dpr, dpr);

    const cx = 110, cy = 110, outerR = 90, innerR = 55;
    ctx.clearRect(0, 0, 220, 220);

    const entries = Object.entries(categoryCounts).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((sum, [, v]) => sum + v, 0);

    if (total === 0) {
      // Empty state ring
      ctx.beginPath();
      ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
      ctx.arc(cx, cy, innerR, 0, Math.PI * 2, true);
      ctx.fillStyle = 'rgba(255,255,255,0.04)';
      ctx.fill();

      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '500 14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No data yet', cx, cy);

      legendContainer.innerHTML = '';
      return;
    }

    let startAngle = -Math.PI / 2;
    const gap = 0.03; // Gap between slices

    for (const [catId, count] of entries) {
      const cat = CATEGORIES[catId] || CATEGORIES.general;
      const sliceAngle = (count / total) * (Math.PI * 2) - gap;

      ctx.beginPath();
      ctx.arc(cx, cy, outerR, startAngle, startAngle + sliceAngle);
      ctx.arc(cx, cy, innerR, startAngle + sliceAngle, startAngle, true);
      ctx.closePath();
      ctx.fillStyle = cat.color;
      ctx.fill();

      startAngle += sliceAngle + gap;
    }

    // Center text
    ctx.fillStyle = '#e8e8f0';
    ctx.font = '700 28px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(total.toString(), cx, cy - 8);
    ctx.fillStyle = 'rgba(232,232,240,0.5)';
    ctx.font = '400 12px Inter, sans-serif';
    ctx.fillText('items', cx, cy + 14);

    // Legend
    legendContainer.innerHTML = '';
    for (const [catId, count] of entries) {
      const cat = CATEGORIES[catId] || CATEGORIES.general;
      const pct = ((count / total) * 100).toFixed(0);
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `
        <span class="legend-color" style="background:${cat.color}"></span>
        <span>${cat.icon} ${cat.name}</span>
        <span class="legend-value">${count} (${pct}%)</span>
      `;
      legendContainer.appendChild(item);
    }
  }

  renderGuideCards() {
    const grid = document.getElementById('guide-grid');
    grid.innerHTML = '';

    for (const [catId, cat] of Object.entries(CATEGORIES)) {
      if (catId === 'general') continue; // Skip general for guide

      const instructions = RECYCLING_INSTRUCTIONS[catId];
      const impact = ENVIRONMENTAL_IMPACT[catId];

      const card = document.createElement('div');
      card.className = 'guide-card';
      card.style.setProperty('--card-gradient', cat.gradient);

      card.innerHTML = `
        <div class="guide-card-header">
          <span class="guide-card-icon">${cat.emoji}</span>
          <span class="guide-card-title">${cat.name}</span>
        </div>
        <p class="guide-card-desc">${cat.description}</p>
        <button class="guide-expand-btn" type="button">
          Learn more <span class="expand-arrow">▼</span>
        </button>
        <div class="guide-card-detail">
          <div class="panel-title" style="margin-top:8px"><span class="panel-icon">♻️</span> How to Recycle</div>
          <ol class="steps-list">
            ${instructions.steps.map((s) => `<li>${s}</li>`).join('')}
          </ol>
          <div class="panel-title" style="margin-top:16px"><span class="panel-icon">💡</span> Tips</div>
          <ul class="tips-list">
            ${instructions.tips.map((t) => `<li>${t}</li>`).join('')}
          </ul>
          <div style="margin-top:16px;font-size:0.85rem;color:var(--text-muted)">
            ⏳ Decomposition: <strong style="color:var(--text-secondary)">${impact.decompositionTime}</strong>
          </div>
        </div>
      `;

      // Expand/collapse
      const btn = card.querySelector('.guide-expand-btn');
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        card.classList.toggle('expanded');
      });

      card.addEventListener('click', () => {
        card.classList.toggle('expanded');
      });

      grid.appendChild(card);
    }
  }

  renderDecompositionTimeline() {
    const container = document.getElementById('decomposition-timeline');
    container.innerHTML = '';

    // Use log scale for bar widths
    const maxLog = Math.log10(1000001); // glass bottle ~1M years

    for (const item of DECOMPOSITION_TIMELINE) {
      const cat = CATEGORIES[item.category] || CATEGORIES.general;
      const logVal = item.years === Infinity ? maxLog : Math.log10(Math.max(item.years, 0.01));
      const widthPercent = Math.max((logVal / maxLog) * 100, 2);

      const row = document.createElement('div');
      row.className = 'timeline-bar-item';
      row.innerHTML = `
        <span class="timeline-bar-label">${item.item}</span>
        <div class="timeline-bar-track">
          <div class="timeline-bar-fill" style="width:0%;background:${cat.color}" data-width="${widthPercent}"></div>
        </div>
        <span class="timeline-bar-time">${item.time}</span>
      `;
      container.appendChild(row);
    }

    // Animate bars on scroll
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const fills = container.querySelectorAll('.timeline-bar-fill');
            fills.forEach((fill, i) => {
              setTimeout(() => {
                fill.style.width = fill.dataset.width + '%';
              }, i * 80);
            });
            observer.disconnect();
          }
        }
      },
      { threshold: 0.3 }
    );
    observer.observe(container);
  }

  setupFilterChips() {
    const chips = document.querySelectorAll('.filter-chip');
    chips.forEach((chip) => {
      chip.addEventListener('click', () => {
        chips.forEach((c) => c.classList.remove('active'));
        chip.classList.add('active');
        this.renderHistory(chip.dataset.filter);
      });
    });
  }

  setupClearHistory() {
    document.getElementById('btn-clear-history').addEventListener('click', () => {
      this.app.historyManager.clear();
      this.renderHistory('all');
      this.updateDashboard();
      // Reset filter chips
      document.querySelectorAll('.filter-chip').forEach((c) => c.classList.remove('active'));
      document.querySelector('.filter-chip[data-filter="all"]').classList.add('active');
      this.app.showToast('🗑️ History cleared', 'success');
    });
  }
}

// ═══════════════════════════════════════════════════════════════
// 8. ECO TICKER
// ═══════════════════════════════════════════════════════════════
class EcoTicker {
  constructor() {
    this.textEl = document.getElementById('eco-ticker-text');
    this.iconEl = document.getElementById('eco-ticker-icon');
    this.currentIndex = Math.floor(Math.random() * ECO_TIPS.length);
    this.show();
    setInterval(() => this.next(), 8000);
  }

  show() {
    const tip = ECO_TIPS[this.currentIndex];
    const cat = CATEGORIES[tip.category];
    this.textEl.textContent = tip.tip;
    this.iconEl.textContent = cat ? cat.icon : '🌍';
  }

  next() {
    this.currentIndex = (this.currentIndex + 1) % ECO_TIPS.length;
    // Re-trigger animation
    const content = document.getElementById('eco-ticker-content');
    content.style.animation = 'none';
    content.offsetHeight; // force reflow
    content.style.animation = '';
    this.show();
  }
}

// ═══════════════════════════════════════════════════════════════
// ECOBOT WIDGET CONTROLLER
// ═══════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════
class EcoScanApp {
  constructor() {
    this.classificationEngine = new ClassificationEngine();
    this.wasteMapper = new WasteMapper();
    this.historyManager = new HistoryManager();
    this.uiController = new UIController(this);
    this.imageHandler = new ImageHandler(this);
    this.cameraHandler = new CameraHandler(this);
    this.particleSystem = new ParticleSystem('particle-canvas');
    this.ecoTicker = new EcoTicker();
    this.ecoBot = new EcoBotController(this, true);

    this.init();
  }

  init() {
    // Render static sections
    this.uiController.renderGuideCards();
    this.uiController.renderDecompositionTimeline();
    this.uiController.renderHistory('all');
    this.uiController.updateDashboard();
    this.uiController.setupFilterChips();
    this.uiController.setupClearHistory();

    // Preload model in background
    this.classificationEngine.loadModel().then(() => {
      console.log('✅ MobileNet loaded and ready');
    }).catch((err) => {
      console.warn('⚠️ Model preload failed, will retry on first classification:', err);
    });
  }

  async classifyImage(imageElement) {
    this.imageHandler.startScanning();

    try {
      // Ensure model is loaded
      await this.classificationEngine.loadModel();

      // Wait a bit for UX (scanning animation)
      await new Promise((r) => setTimeout(r, 1200));

      // Classify
      const predictions = await this.classificationEngine.classify(imageElement);

      // Map to waste category
      const result = this.wasteMapper.map(predictions);

      // Stop scanning
      this.imageHandler.stopScanning();

      // Create thumbnail for history
      const thumbCanvas = document.createElement('canvas');
      thumbCanvas.width = 80;
      thumbCanvas.height = 80;
      const tCtx = thumbCanvas.getContext('2d');
      const size = Math.min(imageElement.naturalWidth, imageElement.naturalHeight);
      const sx = (imageElement.naturalWidth - size) / 2;
      const sy = (imageElement.naturalHeight - size) / 2;
      tCtx.drawImage(imageElement, sx, sy, size, size, 0, 0, 80, 80);

      // Add to history
      this.historyManager.add({
        itemName: result.itemName,
        category: result.category,
        confidence: result.confidence,
        thumbnail: thumbCanvas.toDataURL('image/jpeg', 0.7),
      });

      // Update UI
      this.uiController.showResults(result);
      this.uiController.renderHistory('all');
      this.uiController.updateDashboard();

      // Reset filter
      document.querySelectorAll('.filter-chip').forEach((c) => c.classList.remove('active'));
      document.querySelector('.filter-chip[data-filter="all"]').classList.add('active');

      this.showToast(`${CATEGORIES[result.category].icon} Classified as ${CATEGORIES[result.category].name}!`, 'success');

    } catch (err) {
      console.error('Classification error:', err);
      this.imageHandler.stopScanning();
      this.showToast('❌ Classification failed. Please try again.', 'error');
    }
  }

  showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const icon = document.getElementById('toast-icon');
    const text = document.getElementById('toast-text');

    icon.textContent = type === 'success' ? '✅' : '⚠️';
    text.textContent = message;
    toast.classList.add('active');

    setTimeout(() => {
      toast.classList.remove('active');
    }, 3500);
  }
}

// ─── Bootstrap ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  window.ecoscanApp = new EcoScanApp();
});
