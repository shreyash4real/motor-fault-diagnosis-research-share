// Cinematic scroll-driven background canvas
// Renders motor cross-section → STFT spectrogram → 3-phase traveling sine waves → results
// Crossfades based on scroll progress through 4 hero sections.
//
// Performance-optimized version:
//   - Delta-time animation (consistent speed at any framerate)
//   - Offscreen ImageData buffer for spectrogram (replaces ~69k fillRect/frame)
//   - Batched path operations for motor geometry
//   - Coarser grid + larger trace step for 3-phase waves
//   - Pre-computed particle seeds for neural field

function CinematicBackground({ progress, sectionIdx, sectionProgress }) {
  const { useEffect, useRef } = React;
  const canvasRef = useRef();
  const rafRef = useRef();
  const stateRef = useRef({ progress: 0, section: 0, sectionProgress: 0, t: 0 });
  const prevTimeRef = useRef(null);

  useEffect(() => {
    stateRef.current.progress = progress;
    stateRef.current.section = sectionIdx;
    stateRef.current.sectionProgress = sectionProgress;
  }, [progress, sectionIdx, sectionProgress]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = window.innerWidth + 'px';
      canvas.style.height = window.innerHeight + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener('resize', resize);

    prevTimeRef.current = null;

    function render(timestamp) {
      // Delta-time: consistent animation speed regardless of frame rate
      if (prevTimeRef.current === null) prevTimeRef.current = timestamp;
      const dtMs = timestamp - prevTimeRef.current;
      prevTimeRef.current = timestamp;
      // Clamp delta to avoid huge jumps on tab-switch (cap at ~100ms)
      const dt = Math.min(dtMs, 100) / 1000;

      const W = window.innerWidth, H = window.innerHeight;
      const s = stateRef.current;
      s.t += dt;

      ctx.clearRect(0, 0, W, H);

      const sec = s.section;
      const sp = s.sectionProgress;

      const opacity = (target) => {
        if (sec === target) return 1;
        if (sec === target - 1) return Math.max(0, sp - 0.75) * 4;
        if (sec === target + 1) return Math.max(0, 1 - sp * 4);
        return 0;
      };

      // S0: Motor
      const op0 = opacity(0);
      if (op0 > 0.01) {
        ctx.save();
        ctx.globalAlpha = op0;
        const zoom = sec === 0 ? 1 + sp * 0.6 : 1.6 + (sec > 0 ? 0.4 : 0);
        drawMotor(ctx, W, H, s.t, zoom);
        ctx.restore();
      }

      // S1: STFT spectrogram
      const op1 = opacity(1);
      if (op1 > 0.01) {
        ctx.save();
        ctx.globalAlpha = op1;
        drawSpectrogram(ctx, W, H, s.t, sec === 1 ? sp : 1);
        ctx.restore();
      }

      // S2: 3-phase traveling waves
      const op2 = opacity(2);
      if (op2 > 0.01) {
        ctx.save();
        ctx.globalAlpha = op2;
        drawThreePhase(ctx, W, H, s.t, sec < 2 ? 0 : sec === 2 ? Math.min(1, sp / 0.6) : 1);
        ctx.restore();
      }

      // S3: results — neural mesh / classification field
      const op3 = opacity(3);
      if (op3 > 0.01) {
        ctx.save();
        ctx.globalAlpha = op3;
        drawNeuralField(ctx, W, H, s.t, sec === 3 ? sp : 1);
        ctx.restore();
      }

      rafRef.current = requestAnimationFrame(render);
    }
    rafRef.current = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed', inset: 0,
        width: '100vw', height: '100vh',
        zIndex: 0, pointerEvents: 'none',
      }}
    />
  );
}

// ─── Scene 0: Motor cross-section ───────────────────────────────────────────
// Batched: cooling fins, stator slots, and rotor bars each drawn in a single
// beginPath/stroke or beginPath/fill call instead of one per element.
function drawMotor(ctx, W, H, t, zoom) {
  const cx = W * 0.72, cy = H * 0.5;
  const baseR = Math.min(W, H) * 0.42 * zoom;

  // Background gradient — warm metallic
  const bgGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, baseR * 1.4);
  bgGrad.addColorStop(0, 'rgba(90, 70, 50, 0.18)');
  bgGrad.addColorStop(0.6, 'rgba(40, 32, 25, 0.08)');
  bgGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, W, H);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(t * 0.04);

  // Cooling fins — batch all 64 into one path
  ctx.strokeStyle = 'rgba(184, 67, 31, 0.35)';
  ctx.lineWidth = 2;
  const fins = 64;
  ctx.beginPath();
  for (let i = 0; i < fins; i++) {
    const a = (i / fins) * Math.PI * 2;
    const cosA = Math.cos(a), sinA = Math.sin(a);
    const r1 = baseR * 0.95;
    const r2 = baseR * 1.08;
    ctx.moveTo(cosA * r1, sinA * r1);
    ctx.lineTo(cosA * r2, sinA * r2);
  }
  ctx.stroke();

  // Outer ring (housing)
  ctx.strokeStyle = 'rgba(184, 67, 31, 0.55)';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(0, 0, baseR * 0.95, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = 'rgba(120, 50, 25, 0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(0, 0, baseR * 0.92, 0, Math.PI * 2); ctx.stroke();

  // Stator slots — batch fills and strokes separately
  const slots = 36;

  // Batch slot dark fills
  ctx.fillStyle = 'rgba(20, 15, 10, 0.55)';
  ctx.beginPath();
  for (let i = 0; i < slots; i++) {
    const a = (i / slots) * Math.PI * 2;
    const cosA = Math.cos(a), sinA = Math.sin(a);
    const r1 = baseR * 0.62;
    const r2 = baseR * 0.88;
    const w1 = baseR * 0.04, w2 = baseR * 0.05;
    // Trapezoid corners rotated by angle a
    const p1x = cosA * r1 - sinA * (-w1/2);
    const p1y = sinA * r1 + cosA * (-w1/2);
    const p2x = cosA * r1 - sinA * (w1/2);
    const p2y = sinA * r1 + cosA * (w1/2);
    const p3x = cosA * r2 - sinA * (w2/2);
    const p3y = sinA * r2 + cosA * (w2/2);
    const p4x = cosA * r2 - sinA * (-w2/2);
    const p4y = sinA * r2 + cosA * (-w2/2);
    ctx.moveTo(p1x, p1y);
    ctx.lineTo(p2x, p2y);
    ctx.lineTo(p3x, p3y);
    ctx.lineTo(p4x, p4y);
    ctx.closePath();
  }
  ctx.fill();

  // Batch slot strokes
  ctx.strokeStyle = 'rgba(184, 67, 31, 0.3)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  for (let i = 0; i < slots; i++) {
    const a = (i / slots) * Math.PI * 2;
    const cosA = Math.cos(a), sinA = Math.sin(a);
    const r1 = baseR * 0.62;
    const r2 = baseR * 0.88;
    const w1 = baseR * 0.04, w2 = baseR * 0.05;
    const p1x = cosA * r1 - sinA * (-w1/2);
    const p1y = sinA * r1 + cosA * (-w1/2);
    const p2x = cosA * r1 - sinA * (w1/2);
    const p2y = sinA * r1 + cosA * (w1/2);
    const p3x = cosA * r2 - sinA * (w2/2);
    const p3y = sinA * r2 + cosA * (w2/2);
    const p4x = cosA * r2 - sinA * (-w2/2);
    const p4y = sinA * r2 + cosA * (-w2/2);
    ctx.moveTo(p1x, p1y);
    ctx.lineTo(p2x, p2y);
    ctx.lineTo(p3x, p3y);
    ctx.lineTo(p4x, p4y);
    ctx.closePath();
  }
  ctx.stroke();

  // Copper winding hints — batch
  ctx.fillStyle = 'rgba(217, 119, 68, 0.25)';
  for (let i = 0; i < slots; i++) {
    const a = (i / slots) * Math.PI * 2;
    ctx.save();
    ctx.rotate(a);
    const w1 = baseR * 0.04;
    const r1 = baseR * 0.62;
    const r2 = baseR * 0.88;
    ctx.fillRect(-w1/2 * 0.6, r1 + 4, w1 * 0.6, (r2 - r1) * 0.85);
    ctx.restore();
  }

  // Air gap
  ctx.strokeStyle = 'rgba(40, 32, 25, 0.6)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(0, 0, baseR * 0.6, 0, Math.PI * 2); ctx.stroke();

  // Rotor body
  const rotorGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, baseR * 0.55);
  rotorGrad.addColorStop(0, 'rgba(50, 40, 30, 0.7)');
  rotorGrad.addColorStop(0.7, 'rgba(30, 22, 18, 0.6)');
  rotorGrad.addColorStop(1, 'rgba(20, 14, 10, 0.7)');
  ctx.fillStyle = rotorGrad;
  ctx.beginPath(); ctx.arc(0, 0, baseR * 0.55, 0, Math.PI * 2); ctx.fill();

  // Rotor bars — batch fill + stroke
  const bars = 28;
  ctx.rotate(t * 0.6);
  const barR = baseR * 0.5;
  const barSize = baseR * 0.025;

  ctx.fillStyle = 'rgba(217, 119, 68, 0.55)';
  ctx.beginPath();
  for (let i = 0; i < bars; i++) {
    const a = (i / bars) * Math.PI * 2;
    const x = Math.cos(a) * barR, y = Math.sin(a) * barR;
    ctx.moveTo(x + barSize, y);
    ctx.arc(x, y, barSize, 0, Math.PI * 2);
  }
  ctx.fill();

  ctx.strokeStyle = 'rgba(120, 60, 30, 0.6)';
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  for (let i = 0; i < bars; i++) {
    const a = (i / bars) * Math.PI * 2;
    const x = Math.cos(a) * barR, y = Math.sin(a) * barR;
    ctx.moveTo(x + barSize, y);
    ctx.arc(x, y, barSize, 0, Math.PI * 2);
  }
  ctx.stroke();

  // Shaft
  ctx.fillStyle = 'rgba(60, 50, 40, 0.85)';
  ctx.beginPath(); ctx.arc(0, 0, baseR * 0.12, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = 'rgba(180, 150, 120, 0.6)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(0, 0, baseR * 0.12, 0, Math.PI * 2); ctx.stroke();
  // Keyway
  ctx.fillStyle = 'rgba(20, 14, 10, 0.85)';
  ctx.fillRect(-baseR * 0.015, -baseR * 0.13, baseR * 0.03, baseR * 0.04);

  ctx.restore();

  // Subtle vignette
  const vig = ctx.createRadialGradient(W*0.5, H*0.5, 0, W*0.5, H*0.5, Math.max(W, H) * 0.7);
  vig.addColorStop(0, 'rgba(0,0,0,0)');
  vig.addColorStop(1, 'rgba(0,0,0,0.45)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);
}

// ─── Viridis LUT — precomputed 256-entry lookup table ───────────────────────
// Built once, shared by drawSpectrogram.  Stores [r, g, b] per entry.
const _viridisLUT = (function() {
  const stops = [
    [68, 1, 84],    // 0
    [59, 82, 139],  // 0.25
    [33, 144, 141], // 0.5
    [94, 201, 98],  // 0.75
    [253, 231, 37], // 1
  ];
  const N = 256;
  const lut = new Array(N);
  for (let idx = 0; idx < N; idx++) {
    const t = idx / (N - 1);
    const i = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
    const f = t * (stops.length - 1) - i;
    const a = stops[i], b = stops[i + 1];
    lut[idx] = [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ];
  }
  return lut;
})();

// ─── Scene 1: STFT spectrogram ──────────────────────────────────────────────
// Uses putImageData with a reusable buffer instead of per-cell fillRect.
// This replaces ~69 000 fillRect calls with a single pixel-buffer write.
let _spectroImgData = null;
let _spectroW = 0;
let _spectroH = 0;

function drawSpectrogram(ctx, W, H, t, intensity) {
  const cellW = 6, cellH = 5;
  const cols = Math.ceil(W / cellW);
  const rows = Math.ceil(H / cellH);
  const pixW = cols * cellW;
  const pixH = rows * cellH;

  // Reuse ImageData if size hasn't changed
  if (!_spectroImgData || _spectroW !== pixW || _spectroH !== pixH) {
    _spectroImgData = ctx.createImageData(pixW, pixH);
    _spectroW = pixW;
    _spectroH = pixH;
  }
  const data = _spectroImgData.data;

  for (let r = 0; r < rows; r++) {
    const freq = 1 - r / rows;

    // Pre-compute frequency-dependent terms (constant across columns)
    const gauss1 = Math.exp(-Math.pow((freq - 0.78) * 30, 2)) * 0.9;
    const gauss2 = Math.exp(-Math.pow((freq - 0.55) * 30, 2)) * 0.55;
    const gauss3 = Math.exp(-Math.pow((freq - 0.32) * 25, 2)) * 0.4;
    const gauss4base = Math.exp(-Math.pow((freq - 0.72) * 60, 2)) * 0.35;
    const noiseR = r * 0.07;
    const noiseR2 = r * 0.13;

    for (let c = 0; c < cols; c++) {
      let v = gauss1 + gauss2 + gauss3;
      // Sideband
      v += gauss4base * (0.5 + 0.5 * Math.sin(c * 0.05 + t * 1.5));
      // Time-varying noise
      const n = Math.sin(c * 0.13 + t * 0.6 + noiseR) * 0.5
              + Math.sin(c * 0.31 - t * 0.4 + noiseR2) * 0.3
              + Math.sin(c * 0.07 + t * 1.2) * 0.2;
      v += n * 0.18;
      v += Math.sin(c * 0.08 - t * 4) * 0.08;

      v = Math.max(0, Math.min(1, v * intensity));

      // Look up color from precomputed LUT
      const lutIdx = (v * 255 + 0.5) | 0; // fast round
      const rgb = _viridisLUT[Math.min(255, lutIdx)];

      // Fill the cell's pixels in the ImageData buffer
      const startPx = r * cellH;
      const endPx = startPx + cellH;
      const startCx = c * cellW;
      const endCx = startCx + cellW;
      for (let py = startPx; py < endPx; py++) {
        const rowOff = py * pixW;
        for (let px = startCx; px < endCx; px++) {
          const off = (rowOff + px) * 4;
          data[off]     = rgb[0];
          data[off + 1] = rgb[1];
          data[off + 2] = rgb[2];
          data[off + 3] = 255;
        }
      }
    }
  }

  ctx.putImageData(_spectroImgData, 0, 0);

  // Top overlay
  ctx.fillStyle = 'rgba(15, 10, 25, 0.4)';
  ctx.fillRect(0, 0, W, H);

  // Vignette
  const vig = ctx.createLinearGradient(0, 0, 0, H);
  vig.addColorStop(0, 'rgba(8, 5, 15, 0.5)');
  vig.addColorStop(0.5, 'rgba(0, 0, 0, 0)');
  vig.addColorStop(1, 'rgba(8, 5, 15, 0.6)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);
}

// ─── Scene 2: 3-phase traveling sine waves ──────────────────────────────────
// Optimized: coarser grid, larger trace step (3px vs 2px), batch grid lines.
function drawThreePhase(ctx, W, H, t, intensity) {
  ctx.fillStyle = 'rgba(18, 16, 14, 0.92)';
  ctx.fillRect(0, 0, W, H);

  // Grid — batch all lines into one path
  const gridStep = 60;
  ctx.strokeStyle = 'rgba(217, 119, 68, 0.06)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < W; x += gridStep) {
    ctx.moveTo(x, 0); ctx.lineTo(x, H);
  }
  for (let y = 0; y < H; y += gridStep) {
    ctx.moveTo(0, y); ctx.lineTo(W, y);
  }
  ctx.stroke();

  // Major axis
  ctx.strokeStyle = 'rgba(217, 119, 68, 0.18)';
  ctx.lineWidth = 1;
  const midY = H * 0.5;
  ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(W, midY); ctx.stroke();

  const phases = [
    { color: '#d62020', offset: 0,             label: 'I_a' },
    { color: '#5b9bbf', offset: -2*Math.PI/3,  label: 'I_b' },
    { color: '#a8a040', offset:  2*Math.PI/3,  label: 'I_c' },
  ];
  const amp = H * 0.22;
  const freq = 0.012;
  const speed = 1.8;

  const drawTo = Math.max(0, Math.min(W, W * intensity));
  if (drawTo < 2) return;

  // Pen-down marker
  ctx.strokeStyle = 'rgba(241, 237, 228, 0.10)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(drawTo, 0); ctx.lineTo(drawTo, H); ctx.stroke();

  // Trace step: 3px instead of 2px — ~33% fewer lineTo calls
  const step = 3;
  phases.forEach((p) => {
    ctx.shadowColor = p.color;
    ctx.shadowBlur = 18;
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let started = false;
    for (let x = 0; x <= drawTo; x += step) {
      const mod = Math.sin(x * 0.003 + t * 0.5) * 8;
      const y = midY + Math.sin(x * freq - t * speed + p.offset) * amp + mod;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  });

  // Leading-edge dots
  phases.forEach((p) => {
    const x = drawTo;
    const mod = Math.sin(x * 0.003 + t * 0.5) * 8;
    const y = midY + Math.sin(x * freq - t * speed + p.offset) * amp + mod;
    ctx.fillStyle = p.color;
    ctx.shadowColor = p.color; ctx.shadowBlur = 14;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  });

  // Vignette
  const vig = ctx.createRadialGradient(W*0.5, H*0.5, Math.min(W,H)*0.2, W*0.5, H*0.5, Math.max(W, H) * 0.7);
  vig.addColorStop(0, 'rgba(0,0,0,0)');
  vig.addColorStop(1, 'rgba(8, 6, 4, 0.7)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);
}

// ─── Scene 3: Neural classification field — particle clusters ───────────────
// Optimized: batch contour lines into one path, coarser angle step,
// batch particles per cluster into one path.
function drawNeuralField(ctx, W, H, t, intensity) {
  ctx.fillStyle = 'rgba(241, 237, 228, 0.96)';
  ctx.fillRect(0, 0, W, H);

  // Contour lines — batch all into one path, coarser step (0.08 vs 0.05)
  ctx.strokeStyle = 'rgba(30, 58, 138, 0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let r = 80; r < Math.max(W, H); r += 90) {
    let first = true;
    for (let a = 0; a <= Math.PI * 2 + 0.1; a += 0.08) {
      const wob = Math.sin(a * 4 + t * 0.3 + r * 0.02) * 30;
      const x = W * 0.5 + Math.cos(a) * (r + wob) * 1.4;
      const y = H * 0.5 + Math.sin(a) * (r + wob);
      if (first) { ctx.moveTo(x, y); first = false; }
      else ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // Particle clusters — batch all particles of each cluster into one path
  const clusters = [
    { x: W * 0.22, y: H * 0.30, color: '#1e3a8a', n: 60 },
    { x: W * 0.78, y: H * 0.32, color: '#b8431f', n: 30 },
    { x: W * 0.25, y: H * 0.72, color: '#0369a1', n: 35 },
    { x: W * 0.75, y: H * 0.70, color: '#a16207', n: 22 },
  ];

  clusters.forEach((c, ci) => {
    ctx.fillStyle = c.color + '55';
    ctx.beginPath();
    for (let i = 0; i < c.n; i++) {
      const seed = ci * 1000 + i;
      const ang = (seed * 13.7 + t * 0.05) % (Math.PI * 2);
      const r = (Math.sin(seed * 0.7) * 0.5 + 0.5) * 110 + 20;
      const drift = Math.sin(t * 0.4 + seed) * 6;
      const x = c.x + Math.cos(ang) * r + drift;
      const y = c.y + Math.sin(ang) * r + Math.cos(t * 0.3 + seed) * 6;
      const radius = 2 + (Math.sin(seed) * 0.5 + 0.5) * 2;
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, Math.PI * 2);
    }
    ctx.fill();
  });

  // Top vignette
  const vig = ctx.createLinearGradient(0, 0, 0, H);
  vig.addColorStop(0, 'rgba(241, 237, 228, 0.6)');
  vig.addColorStop(0.5, 'rgba(241, 237, 228, 0.85)');
  vig.addColorStop(1, 'rgba(241, 237, 228, 0.6)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);
}

Object.assign(window, { CinematicBackground });
