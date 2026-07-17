// Cinematic scroll-driven background canvas
// Renders motor cross-section → STFT spectrogram → 3-phase traveling sine waves → results
// Crossfades based on scroll progress through 4 hero sections.

function CinematicBackground({ progress, sectionIdx, sectionProgress }) {
  const { useEffect, useRef } = React;
  const canvasRef = useRef();
  const rafRef = useRef();
  const stateRef = useRef({ progress: 0, section: 0, sectionProgress: 0, t: 0 });

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

    function render() {
      const W = window.innerWidth, H = window.innerHeight;
      const s = stateRef.current;
      s.t += 1/60;

      ctx.clearRect(0, 0, W, H);

      // Section 0: Hero / Upload — motor cross-section
      // Section 1: Configure — STFT spectrogram zoom
      // Section 2: Processing — 3-phase traveling waves
      // Section 3: Results — clean grid

      // Render layered, with crossfade between sections
      const sec = s.section;
      const sp = s.sectionProgress;

      // Each scene renders with its own opacity based on which section is active.
      // Crossfades happen in the last 25 % of the previous section / first 25 % of
      // the next, so the wave-trace finish at sp≈0.6 gets a beat of full visibility
      // before the results scene fades in.
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
        // Zoom factor: in section 0, slight zoom; in transition out, zoom in dramatically
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
        // Wave completes by sp≈0.6 so user sees the full trace before the
        // results crossfade begins at sp=0.75.
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
    render();
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

// ─── Scene 0: Motor cross-section, drawn with concentric SVG-style geometry ─
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

  // Outer housing — rust/iron
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(t * 0.04);

  // Cooling fins (radial bars)
  ctx.strokeStyle = 'rgba(184, 67, 31, 0.35)';
  ctx.lineWidth = 2;
  const fins = 64;
  for (let i = 0; i < fins; i++) {
    const a = (i / fins) * Math.PI * 2;
    const r1 = baseR * 0.95;
    const r2 = baseR * 1.08;
    ctx.beginPath();
    ctx.moveTo(Math.cos(a) * r1, Math.sin(a) * r1);
    ctx.lineTo(Math.cos(a) * r2, Math.sin(a) * r2);
    ctx.stroke();
  }

  // Outer ring (housing)
  ctx.strokeStyle = 'rgba(184, 67, 31, 0.55)';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(0, 0, baseR * 0.95, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = 'rgba(120, 50, 25, 0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(0, 0, baseR * 0.92, 0, Math.PI * 2); ctx.stroke();

  // Stator slots (laminated)
  const slots = 36;
  for (let i = 0; i < slots; i++) {
    const a = (i / slots) * Math.PI * 2;
    const r1 = baseR * 0.62;
    const r2 = baseR * 0.88;
    ctx.save();
    ctx.rotate(a);
    // Slot — trapezoid
    ctx.beginPath();
    const w1 = baseR * 0.04, w2 = baseR * 0.05;
    ctx.moveTo(-w1/2, r1); ctx.lineTo(w1/2, r1);
    ctx.lineTo(w2/2, r2); ctx.lineTo(-w2/2, r2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(20, 15, 10, 0.55)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(184, 67, 31, 0.3)';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Copper winding hint
    ctx.fillStyle = 'rgba(217, 119, 68, 0.25)';
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

  // Rotor bars (these are what break in "broken rotor bar" fault!)
  const bars = 28;
  ctx.rotate(t * 0.6); // rotor spins faster
  for (let i = 0; i < bars; i++) {
    const a = (i / bars) * Math.PI * 2;
    const r = baseR * 0.5;
    const x = Math.cos(a) * r, y = Math.sin(a) * r;
    ctx.fillStyle = 'rgba(217, 119, 68, 0.55)';
    ctx.beginPath(); ctx.arc(x, y, baseR * 0.025, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(120, 60, 30, 0.6)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

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

// ─── Scene 1: STFT spectrogram — vertical frequency stripes scrolling ───────
function drawSpectrogram(ctx, W, H, t, intensity) {
  // Build a procedural spectrogram-ish field: time on x, frequency on y
  // Use noise-like striping with horizontal bands at certain frequencies (fault signatures)
  const cellW = 6, cellH = 5;
  const cols = Math.ceil(W / cellW);
  const rows = Math.ceil(H / cellH);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cellW;
      const y = r * cellH;
      const freq = 1 - r / rows; // higher row = lower frequency

      // Base spectral envelope — peaks at fundamental (50Hz) and harmonics
      let v = 0;
      // Fundamental band
      v += Math.exp(-Math.pow((freq - 0.78) * 30, 2)) * 0.9;
      // 3rd harmonic
      v += Math.exp(-Math.pow((freq - 0.55) * 30, 2)) * 0.55;
      // 5th harmonic
      v += Math.exp(-Math.pow((freq - 0.32) * 25, 2)) * 0.4;
      // Sideband (fault indicator)
      v += Math.exp(-Math.pow((freq - 0.72) * 60, 2)) * 0.35 * (0.5 + 0.5 * Math.sin(c * 0.05 + t * 1.5));
      // Time-varying noise
      const n = Math.sin(c * 0.13 + t * 0.6 + r * 0.07) * 0.5
              + Math.sin(c * 0.31 - t * 0.4 + r * 0.13) * 0.3
              + Math.sin(c * 0.07 + t * 1.2) * 0.2;
      v += n * 0.18;
      // Scrolling time
      v += Math.sin(c * 0.08 - t * 4) * 0.08;

      v = Math.max(0, Math.min(1, v * intensity));

      // Viridis-ish colormap (purple → blue → green → yellow)
      const color = viridis(v);
      ctx.fillStyle = color;
      ctx.fillRect(x, y, cellW, cellH);
    }
  }

  // Top axis label region
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

function viridis(t) {
  // simplified viridis approximation
  const stops = [
    [68, 1, 84],    // 0
    [59, 82, 139],  // 0.25
    [33, 144, 141], // 0.5
    [94, 201, 98],  // 0.75
    [253, 231, 37], // 1
  ];
  const i = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
  const f = t * (stops.length - 1) - i;
  const a = stops[i], b = stops[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bb = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bb})`;
}

// ─── Scene 2: 3-phase traveling sine waves ──────────────────────────────────
function drawThreePhase(ctx, W, H, t, intensity) {
  // Background — warm dark
  ctx.fillStyle = 'rgba(18, 16, 14, 0.92)';
  ctx.fillRect(0, 0, W, H);

  // Grid (oscilloscope)
  ctx.strokeStyle = 'rgba(217, 119, 68, 0.06)';
  ctx.lineWidth = 1;
  const gridStep = 60;
  for (let x = 0; x < W; x += gridStep) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y < H; y += gridStep) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  // Major axes
  ctx.strokeStyle = 'rgba(217, 119, 68, 0.18)';
  ctx.lineWidth = 1;
  const midY = H * 0.5;
  ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(W, midY); ctx.stroke();

  // Three phases: Ia, Ib, Ic — 120° apart. I_a is strong red; I_b cobalt; I_c amber.
  const phases = [
    { color: '#d62020', offset: 0,             label: 'I_a' },
    { color: '#5b9bbf', offset: -2*Math.PI/3,  label: 'I_b' },
    { color: '#a8a040', offset:  2*Math.PI/3,  label: 'I_c' },
  ];
  const amp = H * 0.22;
  const freq = 0.012; // spatial frequency
  const speed = 1.8;

  // Signal "draws on" left-to-right as the section progresses. intensity ∈ [0,1].
  // We clip to a moving right edge — wave is absent at top of section, fully drawn
  // only when the user has scrolled to the bottom of the processing section.
  const drawTo = Math.max(0, Math.min(W, W * intensity));
  if (drawTo < 2) return; // nothing yet — leave grid only

  // Pen-down marker — vertical guide at the leading edge of the trace
  ctx.strokeStyle = 'rgba(241, 237, 228, 0.10)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(drawTo, 0); ctx.lineTo(drawTo, H); ctx.stroke();

  phases.forEach((p) => {
    // Glow
    ctx.shadowColor = p.color;
    ctx.shadowBlur = 18;
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let started = false;
    for (let x = 0; x <= drawTo; x += 2) {
      const mod = Math.sin(x * 0.003 + t * 0.5) * 8;
      const y = midY + Math.sin(x * freq - t * speed + p.offset) * amp + mod;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  });

  // Leading-edge sample dots — pinned to the right tip of each trace
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
function drawNeuralField(ctx, W, H, t, intensity) {
  // Soft warm paper background to match results aesthetic
  ctx.fillStyle = 'rgba(241, 237, 228, 0.96)';
  ctx.fillRect(0, 0, W, H);

  // Faint contour lines suggesting decision boundary / probability field
  ctx.strokeStyle = 'rgba(30, 58, 138, 0.08)';
  ctx.lineWidth = 1;
  for (let r = 80; r < Math.max(W, H); r += 90) {
    ctx.beginPath();
    for (let a = 0; a <= Math.PI * 2 + 0.1; a += 0.05) {
      const wob = Math.sin(a * 4 + t * 0.3 + r * 0.02) * 30;
      const x = W * 0.5 + Math.cos(a) * (r + wob) * 1.4;
      const y = H * 0.5 + Math.sin(a) * (r + wob);
      if (a === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Particle clusters — 4 classes at 4 anchor points
  const clusters = [
    { x: W * 0.22, y: H * 0.30, color: '#1e3a8a', n: 60 },  // healthy
    { x: W * 0.78, y: H * 0.32, color: '#b8431f', n: 30 },  // stator
    { x: W * 0.25, y: H * 0.72, color: '#0369a1', n: 35 },  // bearing
    { x: W * 0.75, y: H * 0.70, color: '#a16207', n: 22 },  // rotor
  ];

  clusters.forEach((c, ci) => {
    for (let i = 0; i < c.n; i++) {
      const seed = ci * 1000 + i;
      const ang = (seed * 13.7 + t * 0.05) % (Math.PI * 2);
      const r = (Math.sin(seed * 0.7) * 0.5 + 0.5) * 110 + 20;
      const drift = Math.sin(t * 0.4 + seed) * 6;
      const x = c.x + Math.cos(ang) * r + drift;
      const y = c.y + Math.sin(ang) * r + Math.cos(t * 0.3 + seed) * 6;
      ctx.fillStyle = c.color + '55';
      ctx.beginPath(); ctx.arc(x, y, 2 + (Math.sin(seed) * 0.5 + 0.5) * 2, 0, Math.PI * 2); ctx.fill();
    }
  });

  // Top vignette to fade content area
  const vig = ctx.createLinearGradient(0, 0, 0, H);
  vig.addColorStop(0, 'rgba(241, 237, 228, 0.6)');
  vig.addColorStop(0.5, 'rgba(241, 237, 228, 0.85)');
  vig.addColorStop(1, 'rgba(241, 237, 228, 0.6)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);
}

Object.assign(window, { CinematicBackground });
