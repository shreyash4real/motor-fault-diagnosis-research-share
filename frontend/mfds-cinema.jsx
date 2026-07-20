// Cinematic scroll-driven Motor Fault Detection app
const { useState, useEffect, useRef, useMemo } = React;

// ─── Reference case and pipeline profile ────────────────────────────────────
// SAMPLE_DATASET is a 3-phase reference recording. The artifact-backed results
// below are loaded from frontend/data/results-data.json.
const SAMPLE_DATASET = { count: 3, size: '91.4 MB', duration: '15 s · 300 000 samples', channels: 'I_a · I_b · I_c' };

const STAGES = [
  { name: 'Manifest validation',  desc: '3 CSVs · row-count · NaN · clipping · RMS audit',           time: '00:00:02' },
  { name: 'Bandpass denoise',     desc: '5 – 5 000 Hz · 4th-order Butterworth · zero-phase filtfilt', time: '00:00:09' },
  { name: 'Segmentation',         desc: '1.0 s window · 0.25 s stride · 75 % overlap · 57 seg/col',   time: '00:00:03' },
  { name: 'Linear STFT',          desc: 'Hann · nperseg 1024 · noverlap 896 · 0–3 kHz · 154 × 149',   time: '00:00:14' },
  { name: 'DWT decomposition',    desc: 'db8 · level 10 · cD2…cA10 · per-level z-score',              time: '00:00:08' },
  { name: 'Envelope spectrum',    desc: 'Hilbert · log-magnitude FFT · 0–500 Hz',                     time: '00:00:05' },
  { name: 'Fusion & calibration', desc: 'Soft voting · T = 2.40 / 0.74 / 0.52 · 4-class softmax',     time: '00:00:04' },
];
// Total stage time is 45 s — used by the elapsed counter below.
const TOTAL_STAGE_SEC = 45;

// ─── Scroll progress hook ──────────────────────────────────────────────────
// Sections aren't always exactly 1 viewport tall — content can push them past
// 100vh — so we measure each section's actual position in the document and pick
// the one whose body straddles the viewport midpoint. Progress is how far the
// user has scrolled inside that section, normalized to its real height.
const SECTION_IDS = ['sec-hero', 'sec-configure', 'sec-processing', 'sec-results'];
function useScrollNarrative(numSections) {
  const [state, setState] = useState({ section: 0, sectionProgress: 0, total: 0 });
  useEffect(() => {
    function onScroll() {
      const scrollY = window.scrollY;
      const vh = window.innerHeight;
      const docH = document.documentElement.scrollHeight - vh;
      const total = docH > 0 ? scrollY / docH : 0;

      // Section flips when its top reaches the viewport top — that way the
      // section's full height maps to sectionProgress 0→1 and we never lose the
      // last slice of the crossfade to a premature section index increment.
      const ref = scrollY + 1;
      let section = 0;
      for (let i = 0; i < SECTION_IDS.length; i++) {
        const el = document.getElementById(SECTION_IDS[i]);
        if (el && ref >= el.offsetTop) section = i;
      }
      const el = document.getElementById(SECTION_IDS[section]);
      let sectionProgress = 0;
      if (el) {
        const top = el.offsetTop;
        const h = Math.max(1, el.offsetHeight);
        sectionProgress = Math.max(0, Math.min(1, (scrollY - top) / h));
      }
      setState({ section, sectionProgress, total });
    }
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, [numSections]);
  return state;
}

// ─── Top brand bar ─────────────────────────────────────────────────────────
function TopBar({ activeSection, light }) {
  const txt = light ? 'var(--ink)' : 'var(--paper)';
  const tag = light ? 'var(--ink-3)' : 'rgba(241,237,228,0.55)';
  const amp = light ? 'var(--accent)' : 'var(--accent)';
  return (
    <div className="top-bar">
      <div>
        <div className="brand" style={{ color: txt, mixBlendMode: 'normal' }}>
          Current<span className="ampersand" style={{ color: amp }}>Guard</span>
        </div>
        <div className="brand-tag" style={{ color: tag, mixBlendMode: 'normal' }}>
          Motor-current diagnostics · Prototype · 2026
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.62rem', color: tag, letterSpacing: '0.18em', textTransform: 'uppercase' }}>
          {STEPS[activeSection]?.roman} / {STEPS[activeSection]?.label}
        </span>
      </div>
    </div>
  );
}

// ─── Right step rail ───────────────────────────────────────────────────────
function StepRail({ activeSection, onClick, light }) {
  return (
    <div className={`step-rail ${light ? 'light' : ''}`}>
      {STEPS.map((s, i) => (
        <div
          key={s.id}
          className={`step-rail-item ${i === activeSection ? 'active' : ''}`}
          onClick={() => onClick(i)}
        >
          <span className="step-rail-num">{s.roman}</span>
          <span className="step-rail-bar"></span>
          <span className="step-rail-label">{s.label}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Section 0: Hero / Upload ──────────────────────────────────────────────
function SectionHero({ files, onUpload, goSection }) {
  const [over, setOver] = useState(false);
  return (
    <section className="cinema" id="sec-hero" data-screen-label="01 Upload">
      <div className="hero-wrap">
        <div>
          <div className="hero-eyebrow">Condition monitoring · Industrial induction motors</div>
          <h1 className="hero-title">
            Read the motor<br />
            through its<br />
            <em>current.</em>
          </h1>
          <p className="hero-lede">
            CurrentGuard turns the three phases already present at an induction motor
            into a condition-monitoring signal. This prototype lets a plant team inspect
            the diagnostic evidence behind healthy, stator, bearing, and rotor-fault states.
          </p>
          <div className="hero-meta-grid">
            <div className="hero-meta-item">
              <span className="k">Subject</span>
              <span className="v">5.5 kW IM, 4-pole</span>
            </div>
            <div className="hero-meta-item">
              <span className="k">Sample rate</span>
              <span className="v">20 kHz</span>
            </div>
            <div className="hero-meta-item">
              <span className="k">Classes</span>
              <span className="v">4 fault states</span>
            </div>
          </div>
        </div>

        <div className="glass-panel">
          <div className="glass-panel-header">
            <h2>Open a reference case</h2>
            <span className="step-roman">I.</span>
          </div>

          <div
            className={`dropzone ${over ? 'over' : ''}`}
            onClick={onUpload}
            onDragOver={e => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={e => { e.preventDefault(); setOver(false); onUpload(); }}
          >
            <div className="dropzone-content">
              <div className="dropzone-icon-row">
                <span>·I_a</span><span>·I_b</span><span>·I_c</span><span>·20 kHz</span>
              </div>
              <div className="dropzone-title">Load the verified reference recording</div>
              <div className="dropzone-sub">3-phase current · 15 s · stored evaluation evidence</div>
            </div>
          </div>

          {files ? (
            <div style={{ marginTop: '1.4rem', padding: '1rem 1.1rem', background: 'rgba(184, 67, 31, 0.08)', borderLeft: '2px solid var(--accent)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.78rem' }}>
              <span style={{ color: 'var(--paper)' }}>{files.count} files · {files.size} · {files.duration}</span>
              <span style={{ color: 'rgba(241,237,228,0.55)', fontSize: '0.7rem' }}>{files.channels}</span>
            </div>
          ) : (
            <p style={{ marginTop: '1.4rem', fontSize: '0.78rem', color: 'rgba(241, 237, 228, 0.5)', fontFamily: "'Fraunces', serif", fontStyle: 'italic' }}>
              This public demo uses a committed reference case. Live plant-file ingestion is the next pilot integration.
            </p>
          )}

          <div className="btn-row">
            <button className="btn" onClick={onUpload}>{files ? 'Reload reference case' : 'Load reference case'}</button>
            <button className="btn ghost" onClick={() => goSection(1)}>Inspect methodology ↓</button>
          </div>
        </div>
      </div>

      <div className="scroll-hint">
        Scroll to begin <span className="arrow">↓</span>
      </div>
    </section>
  );
}

// ─── Section 1: Configure ──────────────────────────────────────────────────
// All parameters are fixed by the trained pipeline (see AGENTS.md and MODEL_CARD.md).
// This page is a read-only manifest of the locked acquisition + spectral + inference
// configuration that produced the model weights — not an editable form.
function SectionConfigure({ goSection, model, setModel }) {
  const lockedRowStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr auto',
    alignItems: 'baseline',
    gap: '1rem',
    padding: '0.7rem 0',
    borderBottom: '1px solid var(--glass-rule)',
  };
  const labelStyle = {
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: '0.66rem',
    letterSpacing: '0.18em',
    textTransform: 'uppercase',
    color: 'var(--glass-text-3)',
  };
  const valStyle = {
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: '0.92rem',
    color: 'var(--paper)',
    letterSpacing: '0.02em',
  };
  const lockChip = {
    display: 'inline-block',
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: '0.55rem',
    letterSpacing: '0.18em',
    color: 'var(--glass-text-3)',
    border: '1px solid var(--glass-rule)',
    padding: '0.1rem 0.4rem',
    marginLeft: '0.5rem',
    textTransform: 'uppercase',
  };

  const Row = ({ k, v, hint }) => (
    <div style={lockedRowStyle}>
      <div>
        <div style={labelStyle}>{k}<span style={lockChip}>fixed</span></div>
        {hint && (
          <div style={{
            fontFamily: "'Fraunces', serif", fontStyle: 'italic',
            fontSize: '0.78rem', color: 'var(--glass-text-3)',
            marginTop: '0.25rem', lineHeight: 1.45,
          }}>{hint}</div>
        )}
      </div>
      <div style={valStyle}>{v}</div>
    </div>
  );

  return (
    <section className="cinema" id="sec-configure" data-screen-label="02 Configure">
      <div className="cfg-grid">
        <div className="cfg-header">
          <h2 className="cfg-title">Locked to the<br /><em>trained pipeline.</em></h2>
          <p className="cfg-subtitle">
            Every parameter below is fixed by the model weights — segmentation,
            denoising, line frequency, pole count. Touching any of them invalidates
            the inference graph, so we don't.
          </p>
        </div>

        <div className="cfg-panels">
          <div className="cfg-panel">
            <span className="h3-num">II.A · Acquisition</span>
            <h3>Signal &amp; machine</h3>
            <div className="cfg-panel-divider"></div>
            <div>
              <Row k="Sample rate" v="20 000 Hz" hint="Dataset standard, 20 kS/s per channel." />
              <Row k="Poles" v="4" hint="Two pole pairs · 5.5 kW induction motor." />
              <Row k="Line frequency" v="50 Hz" hint="Mains carrier — stripped during envelope analysis." />
              <Row k="Channels" v="3 · I_a · I_b · I_c" />
              <Row k="Recording length" v="15 s · 300 000 samples" />
            </div>
          </div>

          <div className="cfg-panel">
            <span className="h3-num">II.B · Pre-filter &amp; segmentation</span>
            <h3>Bandpass &amp; window</h3>
            <div className="cfg-panel-divider"></div>
            <div>
              <Row k="Bandpass" v="5 – 5 000 Hz" hint="4th-order Butterworth, zero-phase filtfilt." />
              <Row k="Window length" v="20 000 samples · 1.0 s" />
              <Row k="Stride" v="5 000 samples · 0.25 s" />
              <Row k="Overlap" v="75 %" hint="57 segments per recording at this stride." />
              <Row k="STFT" v="nperseg 1024 · noverlap 896 · Hann" hint="Linear STFT, 87.5 % intra-window overlap, 0–3 kHz cutoff → 154 bins." />
            </div>
          </div>

          <div className="cfg-panel" style={{ gridColumn: '1 / -1' }}>
            <span className="h3-num">II.C · Inference</span>
            <h3>Classifier</h3>
            <div className="cfg-panel-divider"></div>
            <p style={{
              fontFamily: "'Fraunces', serif", fontStyle: 'italic',
              fontSize: '0.85rem', color: 'var(--glass-text-3)',
              marginBottom: '1rem', lineHeight: 1.5,
            }}>
              The only operator decision on this page — pick a representation. All three
              feed identical 4-class softmax heads; numbers shown are stored evaluation
              results within the declared current-sensing operating envelope.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.85rem' }}>
              {[
                { k: 'stft-dwt-temperature', name: 'STFT + DWT · temperature', acc: '99.08 %', desc: 'Equal soft vote after validation-set temperature scaling.' },
                { k: 'stft-dwt-validation-f1', name: 'STFT + DWT · validation-F1', acc: '98.55 %', desc: 'Per-class validation-F1 weighted soft vote.' },
                { k: 'stft-dwt-envelope-temperature', name: 'STFT + DWT + Envelope · temperature', acc: '99.85 %', desc: 'Three current-derived views with calibrated equal voting, within the declared scope.' },
                { k: 'stft-dwt-envelope-validation-f1', name: 'STFT + DWT + Envelope · validation-F1', acc: '99.77 %', desc: 'Three current-derived views with per-class F1 weights.' },
              ].map(m => {
                const active = model === m.k;
                return (
                  <div
                    key={m.k}
                    onClick={() => setModel(m.k)}
                    style={{
                      padding: '0.95rem 1rem', position: 'relative',
                      border: active ? '1px solid var(--accent)' : '1px solid var(--glass-rule)',
                      background: active ? 'rgba(184, 67, 31, 0.12)' : 'rgba(241, 237, 228, 0.03)',
                      cursor: 'pointer', transition: 'all 0.15s',
                    }}
                  >
                    {m.recommended && (
                      <span style={{
                        position: 'absolute', top: '0.6rem', right: '0.7rem',
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: '0.55rem', letterSpacing: '0.16em',
                        color: 'var(--accent)', border: '1px solid var(--accent)',
                        padding: '0.1rem 0.4rem',
                      }}>RECOMMENDED</span>
                    )}
                    <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.66rem', color: active ? 'var(--accent)' : 'var(--glass-text-3)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>{m.k}</p>
                    <p style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '1rem', color: 'var(--paper)', marginTop: '0.2rem' }}>{m.name}</p>
                    <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.78rem', color: active ? 'var(--accent)' : 'var(--glass-text-2)', marginTop: '0.4rem' }}>{m.acc}</p>
                    <p style={{ fontFamily: "'Fraunces', serif", fontSize: '0.8rem', color: 'var(--glass-text-3)', marginTop: '0.3rem', lineHeight: 1.45 }}>{m.desc}</p>
                  </div>
                );
              })}
            </div>
            <div style={{ marginTop: '1.5rem', paddingTop: '1.2rem', borderTop: '1px solid var(--glass-rule)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '1rem', marginBottom: '0.8rem' }}>
                <span className="h3-num">Representative feature views</span>
                <span style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '0.78rem', color: 'var(--glass-text-3)' }}>derived from motor current</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.7rem' }}>
                {[
                  ['STFT', 'sample_gallery/stft/healthy_1_speed_50.png', 'time–frequency view'],
                  ['DWT', 'sample_gallery/dwt/healthy_1_speed_50.png', 'wavelet detail view'],
                  ['Envelope', 'sample_gallery/envelope/healthy_1_speed_50.png', 'demodulated spectrum'],
                ].map(([label, src, description]) => (
                  <figure key={label} style={{ margin: 0, background: 'rgba(0, 0, 0, 0.22)', border: '1px solid var(--glass-rule)', overflow: 'hidden' }}>
                    <img src={src} alt={`${label} representation of a healthy motor-current sample`} loading="lazy" style={{ display: 'block', width: '100%', aspectRatio: '1', objectFit: 'cover' }} />
                    <figcaption style={{ padding: '0.55rem 0.65rem' }}>
                      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', color: 'var(--paper)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>{label}</div>
                      <div style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '0.72rem', color: 'var(--glass-text-3)', marginTop: '0.2rem' }}>{description}</div>
                    </figcaption>
                  </figure>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="btn-row" style={{ justifyContent: 'flex-end' }}>
          <button className="btn ghost" onClick={() => goSection(0)}>↑ Back</button>
          <button className="btn" onClick={() => goSection(2)}>Run analysis →</button>
        </div>
      </div>
    </section>
  );
}

// ─── Section 2: Processing ─────────────────────────────────────────────────
function SectionProcessing({ active, goSection }) {
  // Auto-progress on a timer once the section becomes active.
  const [t, setT] = useState(0); // 0..1
  const startedRef = useRef(false);
  useEffect(() => {
    if (!active || startedRef.current) return;
    startedRef.current = true;
    const start = performance.now();
    const DUR = 9000; // 9 s to walk all stages
    let raf;
    const tick = () => {
      const e = (performance.now() - start) / DUR;
      if (e >= 1) { setT(1); return; }
      setT(e); raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);
  const stageProgress = t;
  const stageIdx = Math.min(STAGES.length - 1, Math.floor(stageProgress * STAGES.length));
  const elapsedSec = Math.floor(stageProgress * TOTAL_STAGE_SEC);
  const elapsed = `${String(Math.floor(elapsedSec/60)).padStart(2,'0')}:${String(elapsedSec%60).padStart(2,'0')}`;
  return (
    <section className="cinema" id="sec-processing" data-screen-label="03 Processing">
      <div className="proc-wrap">
        <h2 className="proc-title">Replay the pipeline,<br /><em>inspect the evidence.</em></h2>
        <p className="proc-status">
          <span className="dot"></span>{stageProgress >= 1 ? `Playback complete · ${STAGES.length} of ${STAGES.length}` : `Playback stage ${stageIdx + 1} of ${STAGES.length} · ${STAGES[stageIdx].name}`}
        </p>

        <div className="proc-grid">
          <div className="proc-stages">
            {STAGES.map((s, i) => (
              <div key={i} className={`proc-stage ${i === stageIdx ? 'active' : ''}`}>
                <span className="proc-stage-num">{String(i + 1).padStart(2, '0')}</span>
                <div>
                  <div className="proc-stage-name">{s.name}</div>
                  <div className="proc-stage-desc">{s.desc}</div>
                </div>
                <span className="proc-stage-time">
                  {i < stageIdx ? '✓ done' : i === stageIdx ? s.time : '—'}
                </span>
              </div>
            ))}
          </div>

          <div className="proc-readout">
            <div>
              <h3>Pipeline profile</h3>
              <p className="big-num">{elapsed}<span>m</span></p>
            </div>
            <div>
              <h3>STFT segments produced</h3>
              {/* 57 seg/col × 3 phases = 171 spectrograms for one Ia/Ib/Ic recording */}
              <p className="big-num">{Math.floor(stageProgress * 171).toLocaleString()}</p>
            </div>
            <div>
              <h3>DWT coefficients</h3>
              {/* 10 sub-bands · 10 140 samples per phase per segment */}
              <p className="big-num">{Math.floor(stageProgress * 10140).toLocaleString()}<span>/phase</span></p>
            </div>
            <div style={{ borderTop: '1px solid var(--glass-rule)', paddingTop: '1rem' }}>
              <h3>Next</h3>
              <p style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '1rem', color: 'var(--paper)' }}>
                This is a visual replay of the fixed pipeline used to produce the stored evaluation—not live inference on a newly uploaded file.
              </p>
            </div>
          </div>
        </div>

        <div className="btn-row" style={{ justifyContent: 'flex-end', marginTop: '2rem' }}>
          <button className="btn ghost" onClick={() => goSection(1)}>↑ Reconfigure</button>
          <button className="btn" onClick={() => goSection(3)} disabled={stageProgress < 1} style={stageProgress < 1 ? { opacity: 0.5, cursor: 'not-allowed' } : {}}>
            {stageProgress < 1 ? 'Working…' : 'See results →'}
          </button>
        </div>
      </div>
    </section>
  );
}

// ─── Section 3: Results ────────────────────────────────────────────────────
function SectionResults({ goSection, data, selectedExperiment, setSelectedExperiment }) {
  const [selectedSampleIndex, setSelectedSampleIndex] = useState(null);

  if (!data) {
    return (
      <section className="cinema results-section" id="sec-results" data-screen-label="04 Results">
        <div className="results-wrap">
          <div className="results-eyebrow">Stored evaluation · loading result bundle</div>
          <h2 className="results-title">Preparing the<br /><em>evidence.</em></h2>
          <p className="results-lede">The product flow is ready. Loading the committed motor-current ensemble outputs.</p>
        </div>
      </section>
    );
  }

  const current = data.experiments.find(item => item.id === selectedExperiment) || data.experiments[0];
  const fullSplit = data.source.fullSplit;
  const labelFor = (key) => ({
    healthy: 'Healthy',
    stator_short: 'Stator fault',
    bearing_bpfo: 'Bearing fault',
    broken_rotor_bar: 'Rotor-bar fault',
  }[key] || key);
  const pct = (value, digits = 2) => `${(value * 100).toFixed(digits)} %`;
  const experimentLabel = (experiment) => experiment.fusion === 'temperature'
    ? 'Temperature-calibrated equal vote'
    : 'Validation-F1 weighted vote';
  const usesEnvelope = current.representations.includes('envelope');
  const usesTemperature = current.fusion === 'temperature';
  const selectConfiguration = (envelope, temperature) => {
    const id = `${envelope ? 'stft-dwt-envelope' : 'stft-dwt'}-${temperature ? 'temperature' : 'validation-f1'}`;
    setSelectedExperiment(id);
    setSelectedSampleIndex(null);
  };
  const casebook = current.samples.filter(sample => !sample.correct).concat(
    current.samples.filter(sample => sample.correct).sort((a, b) => a.margin - b.margin)
  ).slice(0, 24);
  const activeSampleIndex = casebook.some(sample => sample.index === selectedSampleIndex)
    ? selectedSampleIndex
    : casebook[0]?.index ?? 0;
  const activeSample = current.samples[activeSampleIndex] || current.samples[0];
  const sampleAt = (experiment, index) => experiment.samples[index] || experiment.samples.find(sample => sample.index === index);

  return (
    <section className="cinema results-section" id="sec-results" data-screen-label="04 Results">
      <div className="results-wrap">
        <div className="results-eyebrow">Precomputed motor-current evaluation · {data.source.testWindows.toLocaleString()} windows · {data.source.split} split</div>
        <h2 className="results-title">A motor that <em>reveals</em><br />its state in current.</h2>
        <p className="results-lede">Stored current-signal evidence — inspect the candidate, not a simulated browser prediction.</p>
        <div style={{ margin: '1.25rem 0 2.2rem', display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0.8rem' }}>
          <div style={{ position: 'relative', overflow: 'hidden', padding: '1rem 1.1rem', background: 'var(--surface)', border: '1px solid var(--rule-soft)' }}>
            <div style={{ position: 'absolute', inset: '0 auto 0 0', width: '4px', background: 'var(--ink-3)' }} />
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.59rem', letterSpacing: '0.13em', textTransform: 'uppercase', color: 'var(--ink-3)' }}>Full split · archived baseline</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.55rem', marginTop: '0.35rem' }}><span style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '2rem' }}>{pct(fullSplit.accuracy)}</span><span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.64rem', color: 'var(--ink-3)' }}>ACCURACY</span></div>
            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.3rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.66rem', color: 'var(--ink-2)' }}><span>{pct(fullSplit.macroF1)} F1</span><span>{fullSplit.testWindows.toLocaleString()} windows</span></div>
            <a href={fullSplit.artifacts.errorAnalysis} target="_blank" rel="noreferrer" style={{ display: 'inline-block', marginTop: '0.75rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.6rem', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Open archive evidence ↗</a>
          </div>
          <div style={{ position: 'relative', overflow: 'hidden', padding: '1rem 1.1rem', background: 'var(--surface)', border: '1px solid var(--rule-soft)' }}>
            <div style={{ position: 'absolute', inset: '0 auto 0 0', width: '4px', background: 'var(--accent)' }} />
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.59rem', letterSpacing: '0.13em', textTransform: 'uppercase', color: 'var(--accent)' }}>Validated current-sensing envelope</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.55rem', marginTop: '0.35rem' }}><span style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '2rem' }}>{pct(current.accuracy)}</span><span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.64rem', color: 'var(--ink-3)' }}>ACCURACY</span></div>
            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.3rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.66rem', color: 'var(--ink-2)' }}><span>{pct(current.macroF1)} F1</span><span>{data.source.testWindows.toLocaleString()} windows</span></div>
            <a href="https://github.com/shreyash4real/motor-fault-diagnosis-research-share/blob/main/MODEL_CARD.md" target="_blank" rel="noreferrer" style={{ display: 'inline-block', marginTop: '0.75rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.6rem', letterSpacing: '0.08em', textTransform: 'uppercase' }}>View scope boundary ↗</a>
          </div>
          <div style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', gap: '0.85rem', padding: '0.75rem 1rem', background: 'var(--surface2)', border: '1px solid var(--rule-soft)' }}>
            <span style={{ flex: '0 0 auto', padding: '0.28rem 0.45rem', color: 'var(--paper)', background: 'var(--accent)', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.62rem', letterSpacing: '0.08em' }}>44 / 45</span>
            <span style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', color: 'var(--ink-2)' }}>full-split errors came from one bearing-fault group at 100% speed</span>
            <span style={{ marginLeft: 'auto', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.6rem', color: 'var(--ink-3)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>current-only → escalate</span>
          </div>
        </div>

        <div className="results-section-block">
          <div className="results-block-header"><h3>Scoped configuration comparison</h3><span className="num">same held-out split · select a run</span></div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.7rem', marginBottom: '0.8rem' }}>
            {[['Representation set', [['STFT + DWT', false], ['STFT + DWT + Envelope', true]], usesEnvelope], ['Fusion method', [['Temperature calibrated', true], ['Validation-F1 weighted', false]], usesTemperature]].map(([label, options, active]) => (
              <div key={label} style={{ padding: '0.7rem 0.8rem', background: 'var(--surface)', border: '1px solid var(--rule-soft)' }}>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.62rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: '0.45rem' }}>{label}</div>
                <div role="tablist" aria-label={label} style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                  {options.map(([optionLabel, optionValue]) => <button key={optionLabel} role="tab" aria-selected={active === optionValue} onClick={() => label === 'Representation set' ? selectConfiguration(optionValue, usesTemperature) : selectConfiguration(usesEnvelope, optionValue)} style={{ flex: '1 1 8rem', padding: '0.65rem 0.7rem', cursor: 'pointer', color: active === optionValue ? 'var(--paper)' : 'var(--ink)', background: active === optionValue ? 'var(--ink)' : 'var(--surface2)', border: active === optionValue ? '1px solid var(--ink)' : '1px solid var(--rule-soft)', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem' }}>{optionLabel}</button>)}
                </div>
              </div>
            ))}
          </div>
          <div style={{ padding: '0.75rem 1rem', marginBottom: '0.8rem', background: 'var(--surface2)', border: '1px solid var(--rule-soft)', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem' }}>Showing <span style={{ color: 'var(--accent)' }}>{usesEnvelope ? 'envelope-enhanced' : 'STFT + DWT'}</span> features with <span style={{ color: 'var(--accent)' }}>{usesTemperature ? 'temperature calibration' : 'validation-F1 weighting'}</span>.</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.7rem' }}>
            {data.experiments.map(experiment => (
              <button key={experiment.id} onClick={() => { setSelectedExperiment(experiment.id); setSelectedSampleIndex(null); }} style={{ textAlign: 'left', padding: '1rem 1.1rem', cursor: 'pointer', color: 'var(--ink)', background: experiment.id === current.id ? 'var(--surface2)' : 'var(--surface)', border: experiment.id === current.id ? '1px solid var(--accent)' : '1px solid var(--rule-soft)' }}>
                <span style={{ display: 'block', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.62rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: experiment.id === current.id ? 'var(--accent)' : 'var(--ink-3)' }}>{experimentLabel(experiment)}</span>
                <span style={{ display: 'block', fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '1rem', marginTop: '0.35rem' }}>{experiment.representations.map(labelFor).join(' + ')}</span>
                <span style={{ display: 'block', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.8rem', marginTop: '0.5rem' }}>{pct(experiment.accuracy)} accuracy · {pct(experiment.macroF1)} macro-F1</span>
              </button>
            ))}
          </div>
        </div>

        <div className="results-headline-row">
          <MetricCard label="Accuracy" value={(current.accuracy * 100).toFixed(2)} sub={`${current.samples.length - current.errors} / ${current.samples.length} windows`} color="#1e3a8a" />
          <MetricCard label="Macro F1" value={(current.macroF1 * 100).toFixed(2)} sub="unweighted mean" color="#0369a1" />
          <MetricCard label="Errors" value={String(current.errors)} sub={`of ${current.samples.length} windows`} color="#b8431f" />
        </div>

        <div className="results-section-block" style={{ padding: '1.15rem 1.25rem', background: 'var(--surface2)', borderLeft: '3px solid var(--accent)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'baseline' }}><span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.62rem', letterSpacing: '0.13em', color: 'var(--accent)', textTransform: 'uppercase' }}>Capability finding</span><span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.58rem', color: 'var(--ink-3)', textTransform: 'uppercase' }}>decision: escalate</span></div>
          <p style={{ marginTop: '0.55rem', fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '1.16rem', lineHeight: 1.35, color: 'var(--ink-2)' }}>A CNN cannot recover a fault signature the current measurement never exposes.</p>
        </div>

        <div className="results-section-block">
          <div className="results-block-header"><h3>Confusion matrix</h3><span className="num">{current.run}</span></div>
          <ConfusionMatrix cm={current.confusion} classKeys={data.classes.map(item => item.key)} maxVal={Math.max(...current.confusion.flat())} />
        </div>

        <div className="results-section-block">
          <div className="results-block-header"><h3>Per-class performance</h3><span className="num">test windows, not independent motors</span></div>
          <div style={{ background: 'var(--surface)', border: '1px solid var(--rule-soft)', padding: '0.4rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '2.2fr 1fr 1fr 1fr 1fr', gap: '0.5rem', padding: '0.6rem 1rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.62rem', letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--ink-3)', borderBottom: '1px solid var(--rule-soft)' }}><div>Class</div><div style={{ textAlign: 'right' }}>Precision</div><div style={{ textAlign: 'right' }}>Recall</div><div style={{ textAlign: 'right' }}>F1</div><div style={{ textAlign: 'right' }}>Support</div></div>
            {current.perClass.map(item => <div key={item.class} style={{ display: 'grid', gridTemplateColumns: '2.2fr 1fr 1fr 1fr 1fr', gap: '0.5rem', padding: '0.75rem 1rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.78rem', borderBottom: '1px solid var(--rule-soft)' }}><div style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '1rem' }}>{labelFor(item.class)}</div><div style={{ textAlign: 'right' }}>{pct(item.precision, 1)}</div><div style={{ textAlign: 'right' }}>{pct(item.recall, 1)}</div><div style={{ textAlign: 'right', color: 'var(--accent)' }}>{pct(item.f1, 1)}</div><div style={{ textAlign: 'right' }}>{item.support}</div></div>)}
          </div>
        </div>

        <div className="results-section-block">
          <div className="results-block-header"><h3>Sample explorer</h3><span className="num">same sample across all four runs</span></div>
          <select className="form-input" value={activeSampleIndex} onChange={event => setSelectedSampleIndex(Number(event.target.value))}>
            {casebook.map(sample => <option key={sample.index} value={sample.index}>#{sample.index} · {sample.correct ? 'near-boundary correct' : 'misclassified'} · {labelFor(sample.trueClass)} · {sample.speedPct}% speed</option>)}
          </select>
          {activeSample && <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div style={{ padding: '1.2rem', background: 'var(--surface)', border: '1px solid var(--rule-soft)' }}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--ink-3)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>Sample #{activeSample.index}</div>
              <h4 style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '1.35rem', margin: '0.45rem 0' }}>{labelFor(activeSample.trueClass)}</h4>
              <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.75rem', color: 'var(--ink-3)' }}>{activeSample.speedPct}% speed · column {activeSample.column} · segment {activeSample.segment}</p>
              <p style={{ marginTop: '1rem', fontFamily: "'Fraunces', serif", lineHeight: 1.5 }}>{activeSample.correct ? 'Correct on this run.' : `Predicted ${labelFor(activeSample.prediction)} at ${pct(activeSample.confidence, 1)} confidence.`}</p>
              {Object.entries(activeSample.probabilities).map(([key, value]) => <div key={key} style={{ marginTop: '0.7rem' }}><div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.68rem' }}><span>{labelFor(key)}</span><span>{pct(value, 1)}</span></div><div style={{ height: '5px', background: 'var(--surface2)', marginTop: '0.25rem' }}><div style={{ width: `${Math.max(1, value * 100)}%`, height: '100%', background: key === activeSample.prediction ? 'var(--accent)' : 'var(--steel)' }} /></div></div>)}
            </div>
            <div style={{ padding: '1.2rem', background: 'var(--surface)', border: '1px solid var(--rule-soft)' }}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--ink-3)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>Representation gallery</div>
              {Object.keys(activeSample.gallery || {}).length ? <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', marginTop: '0.8rem' }}>{Object.entries(activeSample.gallery).map(([representation, path]) => <a key={representation} href={path} target="_blank" rel="noreferrer"><img src={path} alt={`${representation} representation`} style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', border: '1px solid var(--rule-soft)' }} /><span style={{ display: 'block', marginTop: '0.25rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.6rem', textTransform: 'uppercase' }}>{representation}</span></a>)}</div> : <p style={{ marginTop: '1rem', fontFamily: "'Fraunces', serif", lineHeight: 1.5 }}>No exact gallery image is bundled for this test window. The metadata and probabilities remain exact.</p>}
            </div>
          </div>}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.7rem', marginTop: '1rem' }}>{data.experiments.map(experiment => { const sample = sampleAt(experiment, activeSampleIndex); return <div key={experiment.id} style={{ padding: '0.8rem 1rem', background: 'var(--surface)', border: '1px solid var(--rule-soft)', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem' }}><div style={{ color: 'var(--ink-3)', marginBottom: '0.35rem' }}>{experimentLabel(experiment)}</div><div>{labelFor(sample.prediction)} · {pct(sample.confidence, 1)} · {sample.correct ? 'correct' : 'error'}</div></div>; })}</div>
        </div>

        <div className="btn-row" style={{ justifyContent: 'space-between', borderTop: '1px solid var(--rule)', paddingTop: '1.6rem', marginTop: '2.5rem' }}><button className="btn ghost" style={{ color: 'var(--ink)', borderColor: 'var(--rule)' }} onClick={() => goSection(0)}>↑ Run another</button><div style={{ display: 'flex', gap: '0.8rem' }}><a className="btn dark" href="data/results-data.json" download>Download data</a><a className="btn" href="docs/motor_fault_diagnosis_report.md">Read report</a></div></div>
      </div>
    </section>
  );
}

const ATMOSPHERES = {
  foundry: { label: 'Foundry', desc: 'rust · paper · iron', swatch: ['#b8431f', '#1a1612', '#f1ede4'],
    vars: { '--paper': '#f1ede4', '--paper-2': '#e8e2d4', '--ink': '#1a1612', '--ink-2': '#3a342c', '--ink-3': '#6b6258', '--ink-4': '#9a9185', '--rule': 'rgba(26,22,18,0.18)', '--rule-soft': 'rgba(26,22,18,0.10)', '--surface': '#f8f5ed', '--surface2': '#ebe6d8', '--accent': '#b8431f', '--glass-bg': 'rgba(20,16,12,0.72)', '--glass-bg-2': 'rgba(34,28,22,0.78)', '--glass-rule': 'rgba(241,237,228,0.14)', '--glass-rule-2': 'rgba(241,237,228,0.22)', '--glass-text': '#f1ede4', '--glass-text-2': 'rgba(241,237,228,0.78)', '--glass-text-3': 'rgba(241,237,228,0.55)' },
    canvasFilter: 'none', bodyBg: '#0e0a06' },
  ozone: { label: 'Ozone', desc: 'cyan · steel · glacier', swatch: ['#0891b2', '#08161f', '#e7eef3'],
    vars: { '--paper': '#e7eef3', '--paper-2': '#d8e2ea', '--ink': '#08161f', '--ink-2': '#1d3140', '--ink-3': '#52697a', '--ink-4': '#8a9aa6', '--rule': 'rgba(8,22,31,0.20)', '--rule-soft': 'rgba(8,22,31,0.10)', '--surface': '#eff4f7', '--surface2': '#dde7ed', '--accent': '#0891b2', '--glass-bg': 'rgba(8,18,28,0.78)', '--glass-bg-2': 'rgba(14,28,40,0.82)', '--glass-rule': 'rgba(231,238,243,0.14)', '--glass-rule-2': 'rgba(231,238,243,0.24)', '--glass-text': '#e7eef3', '--glass-text-2': 'rgba(231,238,243,0.78)', '--glass-text-3': 'rgba(231,238,243,0.55)' },
    canvasFilter: 'hue-rotate(175deg) saturate(0.65)', bodyBg: '#020a12' },
  bone: { label: 'Bone', desc: 'graphite · stone · ash', swatch: ['#4a4641', '#14110d', '#ece9e2'],
    vars: { '--paper': '#ece9e2', '--paper-2': '#dedad1', '--ink': '#14110d', '--ink-2': '#2e2a25', '--ink-3': '#5e5953', '--ink-4': '#928d86', '--rule': 'rgba(20,17,13,0.20)', '--rule-soft': 'rgba(20,17,13,0.10)', '--surface': '#f3f0e9', '--surface2': '#e2ddd2', '--accent': '#4a4641', '--glass-bg': 'rgba(16,14,12,0.74)', '--glass-bg-2': 'rgba(28,24,21,0.80)', '--glass-rule': 'rgba(236,233,226,0.14)', '--glass-rule-2': 'rgba(236,233,226,0.24)', '--glass-text': '#ece9e2', '--glass-text-2': 'rgba(236,233,226,0.78)', '--glass-text-3': 'rgba(236,233,226,0.55)' },
    canvasFilter: 'grayscale(1) contrast(0.92)', bodyBg: '#0a0907' },
};

const VOICE_CSS = {
  editorial: '',
  technical: `.hero-title,.cfg-title,.proc-title,.results-title{font-family:'JetBrains Mono',monospace!important;font-style:normal!important;font-weight:500!important;letter-spacing:-0.005em!important;text-transform:uppercase;font-size:clamp(2rem,4vw,3.4rem)!important;line-height:1.05!important}
    .glass-panel-header h2,.cfg-panel h3,.results-block-header h3{font-family:'JetBrains Mono',monospace!important;font-style:normal!important;text-transform:uppercase;font-size:1rem!important;letter-spacing:0.06em!important}
    .hero-title em,.cfg-title em,.proc-title em,.results-title em{font-style:normal!important;font-weight:600!important;color:var(--accent)!important}
    .hero-lede,.results-lede,.cfg-subtitle{font-family:'JetBrains Mono',monospace!important;font-style:normal!important;font-size:0.84rem!important;line-height:1.7!important}`,
  manifesto: `.hero-title,.cfg-title,.proc-title,.results-title{font-family:'Fraunces',serif!important;font-weight:600!important;font-size:clamp(4rem,9vw,8rem)!important;line-height:0.92!important;letter-spacing:-0.04em!important;text-transform:uppercase}
    .hero-title em,.cfg-title em,.proc-title em,.results-title em{font-style:italic!important;font-weight:400!important}
    .hero-lede,.results-lede{font-family:'Fraunces',serif!important;font-style:italic!important;font-size:1.35rem!important;line-height:1.4!important}`,
};

function AtmosphereStyle({ atmosphere, voice, cinematography }) {
  const a = ATMOSPHERES[atmosphere] || ATMOSPHERES.foundry;
  const varStr = Object.entries(a.vars).map(([k, v]) => `${k}:${v};`).join('');
  const canvasOpacity = cinematography === 'off' ? 0 : cinematography === 'hushed' ? 0.38 : 1;
  const offCSS = cinematography === 'off' ? `
    section.cinema:not(.results-section){background:var(--paper)}
    section.cinema:not(.results-section) .glass-panel,section.cinema:not(.results-section) .cfg-panel,section.cinema:not(.results-section) .proc-stages,section.cinema:not(.results-section) .proc-stage,section.cinema:not(.results-section) .proc-readout{background:var(--surface)!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important;border-color:var(--rule-soft)!important}
    section.cinema:not(.results-section) .hero-title,section.cinema:not(.results-section) .cfg-title,section.cinema:not(.results-section) .proc-title,section.cinema:not(.results-section) .glass-panel-header h2,section.cinema:not(.results-section) .cfg-panel h3,section.cinema:not(.results-section) .dropzone-title,section.cinema:not(.results-section) .proc-stage-name,section.cinema:not(.results-section) .hero-meta-item .v,section.cinema:not(.results-section) .toggle-label,section.cinema:not(.results-section) .proc-readout .big-num{color:var(--ink)!important}
    section.cinema:not(.results-section) .hero-lede,section.cinema:not(.results-section) .cfg-subtitle{color:var(--ink-2)!important}
    section.cinema:not(.results-section) .hero-eyebrow,section.cinema:not(.results-section) .proc-status,section.cinema:not(.results-section) .form-label,section.cinema:not(.results-section) .form-hint,section.cinema:not(.results-section) .dropzone-sub,section.cinema:not(.results-section) .proc-stage-desc,section.cinema:not(.results-section) .proc-stage-num,section.cinema:not(.results-section) .proc-readout h3,section.cinema:not(.results-section) .scroll-hint{color:var(--ink-3)!important}
    section.cinema:not(.results-section) .form-input,section.cinema:not(.results-section) .dropzone{color:var(--ink)!important;background:var(--paper)!important;border-color:var(--rule)!important}
    .top-bar .brand,.top-bar .brand-tag{mix-blend-mode:normal!important;color:var(--ink)!important}
  ` : '';
  const css = `:root{${varStr}}body{background:${a.bodyBg}!important}.bg-stage{filter:${a.canvasFilter};opacity:${canvasOpacity};transition:opacity 0.4s,filter 0.4s}${offCSS}${VOICE_CSS[voice] || ''}`;
  return <style dangerouslySetInnerHTML={{ __html: css }} />;
}

function App() {
  const [files, setFiles] = useState(null);
  const [model, setModel] = useState('stft-dwt-envelope-temperature');
  const [resultsData, setResultsData] = useState(null);
  const handleUpload = () => setFiles(SAMPLE_DATASET);
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);

  useEffect(() => {
    fetch('data/results-data.json')
      .then(response => response.ok ? response.json() : Promise.reject(new Error('data/results-data.json unavailable')))
      .then(setResultsData)
      .catch(error => console.warn('Stored results unavailable:', error));
  }, []);

  const { section, sectionProgress } = useScrollNarrative(4);
  const totalProgress = (section + sectionProgress) / 4;

  const goTo = (idx) => {
    const ids = ['sec-hero', 'sec-configure', 'sec-processing', 'sec-results'];
    const el = document.getElementById(ids[idx]);
    if (el) {
      const top = el.getBoundingClientRect().top + window.scrollY;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  };

  const lightChrome = section === 3 || t.cinematography === 'off';

  return (
    <>
      <AtmosphereStyle atmosphere={t.atmosphere} voice={t.voice} cinematography={t.cinematography} />
      <div className="bg-stage">
        <CinematicBackground progress={totalProgress} sectionIdx={section} sectionProgress={sectionProgress} />
      </div>
      <div className="reg-mark reg-tl"></div>
      <div className="reg-mark reg-tr"></div>
      <div className="reg-mark reg-bl"></div>
      <div className="reg-mark reg-br"></div>
      <TopBar activeSection={section} light={lightChrome} />
      <StepRail activeSection={section} onClick={goTo} light={lightChrome} />
      <div className="scroll-stage">
        <SectionHero files={files} onUpload={handleUpload} goSection={goTo} />
        <SectionConfigure goSection={goTo} model={model} setModel={setModel} experiments={resultsData?.experiments || []} />
        <SectionProcessing active={section === 2} goSection={goTo} />
        <SectionResults goSection={goTo} data={resultsData} selectedExperiment={model} setSelectedExperiment={setModel} />
      </div>

      <TweaksPanel title="Tweaks">
        <TweakSection label="Atmosphere" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', padding: '0 8px 8px' }}>
          {Object.entries(ATMOSPHERES).map(([k, a]) => (
            <div key={k} onClick={() => setTweak('atmosphere', k)} style={{ cursor: 'pointer', padding: '8px 8px 9px', borderRadius: '6px', background: t.atmosphere === k ? 'rgba(0,0,0,0.06)' : 'transparent', border: t.atmosphere === k ? '1px solid rgba(0,0,0,0.18)' : '1px solid rgba(0,0,0,0.07)', transition: 'all 0.12s' }}>
              <div style={{ display: 'flex', gap: '2px', height: '22px', borderRadius: '3px', overflow: 'hidden', marginBottom: '5px' }}>
                {a.swatch.map((c, i) => (<div key={i} style={{ flex: i === 0 ? 1.2 : 1, background: c }}></div>))}
              </div>
              <div style={{ fontSize: '10.5px', fontWeight: 600, color: '#29261b' }}>{a.label}</div>
              <div style={{ fontSize: '9.5px', color: '#7a7363', marginTop: '1px' }}>{a.desc}</div>
            </div>
          ))}
        </div>
        <TweakSection label="Editorial voice" />
        <TweakRadio label="Tone" value={t.voice} options={['editorial', 'technical', 'manifesto']} onChange={(v) => setTweak('voice', v)} />
        <div style={{ padding: '0 14px 6px', fontSize: '10px', color: '#7a7363', fontStyle: 'italic', lineHeight: 1.4 }}>
          {t.voice === 'editorial' && 'Fraunces display, italic accents, journal cadence.'}
          {t.voice === 'technical' && 'Mono headlines, no italics, lab-notebook feel.'}
          {t.voice === 'manifesto' && 'Oversize serif, all caps, gallery-wall scale.'}
        </div>
        <TweakSection label="Cinematography" />
        <TweakRadio label="Backdrop" value={t.cinematography} options={['full', 'hushed', 'off']} onChange={(v) => setTweak('cinematography', v)} />
        <div style={{ padding: '0 14px 10px', fontSize: '10px', color: '#7a7363', fontStyle: 'italic', lineHeight: 1.4 }}>
          {t.cinematography === 'full' && 'Live motor, spectrogram, traveling currents — full intensity.'}
          {t.cinematography === 'hushed' && 'Scene art dimmed to a watermark.'}
          {t.cinematography === 'off' && 'Strip the cinema. Pure paper-and-ink reading mode.'}
        </div>
      </TweaksPanel>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
