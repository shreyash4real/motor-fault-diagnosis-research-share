// Shared components

const { useState, useEffect, useRef } = React;

const STEPS = [
  { id: 'upload',     label: 'Upload',     roman: 'I'   },
  { id: 'configure',  label: 'Configure',  roman: 'II'  },
  { id: 'processing', label: 'Analysis',   roman: 'III' },
  { id: 'results',    label: 'Results',    roman: 'IV'  },
];
const STEP_INDEX = { upload: 0, configure: 1, processing: 2, results: 3 };

// Conventional ML palette — dark blue → light blue confusion matrix
const CLASS_META = {
  healthy:          { label: 'Healthy',          color: '#1e3a8a', hex: '#1e3a8a', short: 'HLT', glyph: '○' },
  stator_short:     { label: 'Stator Short',     color: '#b8431f', hex: '#b8431f', short: 'STS', glyph: '◐' },
  bearing_bpfo:     { label: 'Bearing BPFO',     color: '#0369a1', hex: '#0369a1', short: 'BPF', glyph: '◑' },
  broken_rotor_bar: { label: 'Broken Rotor Bar', color: '#a16207', hex: '#a16207', short: 'BRB', glyph: '●' },
};

function Card({ title, icon, children, style }) {
  return (
    <div className="card" style={style}>
      {title && (
        <div className="card-header">
          {icon && <span className="card-header-icon">{icon}</span>}
          <span className="card-header-title">{title}</span>
        </div>
      )}
      <div className="card-body">{children}</div>
    </div>
  );
}

function FormField({ label, hint, children }) {
  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      {children}
      {hint && <p className="form-hint">{hint}</p>}
    </div>
  );
}

function NumberInput({ value, onChange, placeholder, unit, readOnly }) {
  return (
    <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
      <input
        type="number"
        className="form-input"
        value={value}
        onChange={e => onChange && onChange(e.target.value)}
        placeholder={placeholder}
        readOnly={readOnly}
        style={readOnly ? { opacity: 0.55, cursor: 'not-allowed', paddingRight: unit ? '3rem' : undefined, background: 'var(--surface2)' } : { paddingRight: unit ? '3rem' : undefined }}
      />
      {unit && <span style={{ position: 'absolute', right: '0.95rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.72rem', color: 'var(--ink-3)', pointerEvents: 'none', letterSpacing: '0.04em' }}>{unit}</span>}
    </div>
  );
}

function Toggle({ value, onChange, label }) {
  return (
    <div className="toggle-wrap">
      <div className={`toggle ${value ? 'on' : ''}`} onClick={() => onChange(!value)}>
        <div className="toggle-knob"></div>
      </div>
      {label && <span className="toggle-label">{label}</span>}
    </div>
  );
}

function ClassTag({ cls }) {
  const meta = CLASS_META[cls];
  if (!meta) return null;
  return (
    <span className="tag" style={{ color: meta.color, borderColor: 'currentColor', background: 'transparent' }}>
      <span style={{ marginRight: '0.4rem', fontSize: '0.78rem', lineHeight: 1 }}>{meta.glyph}</span>
      {meta.label}
    </span>
  );
}

function SectionLabel({ children }) {
  return <div className="section-label">{children}</div>;
}

Object.assign(window, {
  Card, FormField, NumberInput, Toggle, ClassTag, SectionLabel,
  STEPS, STEP_INDEX, CLASS_META,
});

function MetricCard({ label, value, sub, color, formula, breakdown, calc }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--rule-soft)', borderTop: `2px solid ${color}`, borderRadius: '2px', overflow: 'hidden' }}>
      <div onClick={() => setOpen(!open)} style={{ padding: '1.1rem 1.2rem', cursor: 'pointer', position: 'relative' }}>
        <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.62rem', textTransform: 'uppercase', letterSpacing: '0.16em', color: 'var(--ink-3)', marginBottom: '0.65rem' }}>{label}</p>
        <p style={{ fontFamily: "'Fraunces', serif", fontVariationSettings: '"opsz" 144', fontWeight: 400, fontSize: '2.1rem', color, lineHeight: 1, letterSpacing: '-0.02em' }}>{value}</p>
        <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--ink-3)', marginTop: '0.4rem' }}>{sub}</p>
        <span style={{ position: 'absolute', top: '1rem', right: '1rem', color: 'var(--ink-4)', fontSize: '0.75rem', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s' }}>▾</span>
      </div>
      {open && (
        <div style={{ padding: '1rem 1.2rem', background: 'var(--surface2)', borderTop: '1px solid var(--rule-soft)', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.76rem' }}>
          <p style={{ fontSize: '0.6rem', color: 'var(--ink-3)', marginBottom: '0.5rem', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Formula</p>
          <p style={{ color: 'var(--ink)', marginBottom: '0.85rem', lineHeight: 1.6 }}>{formula}</p>
          {breakdown && breakdown.length > 0 && (
            <>
              <p style={{ fontSize: '0.6rem', color: 'var(--ink-3)', marginBottom: '0.5rem', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Components</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginBottom: '0.85rem' }}>
                {breakdown.map((b, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--ink-3)', borderBottom: '1px dotted var(--rule-soft)', paddingBottom: '0.2rem' }}>
                    <span>{b.k}</span><span style={{ color: 'var(--ink)' }}>{b.v}</span>
                  </div>
                ))}
              </div>
            </>
          )}
          <p style={{ fontSize: '0.6rem', color: 'var(--ink-3)', marginBottom: '0.5rem', letterSpacing: '0.16em', textTransform: 'uppercase' }}>Calculation</p>
          <p style={{ color: color, lineHeight: 1.6, wordBreak: 'break-word', fontWeight: 500 }}>{calc}</p>
        </div>
      )}
    </div>
  );
}

function ExpandableMetricRow({ p, fmt }) {
  const [open, setOpen] = useState(false);
  const m = CLASS_META[p.cls];
  return (
    <div style={{ borderRadius: '2px', background: open ? 'var(--surface2)' : 'transparent', border: open ? '1px solid var(--rule-soft)' : '1px solid transparent' }}>
      <div onClick={() => setOpen(!open)} style={{ display: 'grid', gridTemplateColumns: '2.2fr 1fr 1fr 1fr 1fr 32px', gap: '0.5rem', padding: '0.7rem 1rem', cursor: 'pointer', alignItems: 'center', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.85rem', borderBottom: open ? '1px solid var(--rule-soft)' : '1px solid transparent' }}>
        <div><ClassTag cls={p.cls} /></div>
        <div style={{ textAlign: 'right', color: 'var(--ink)' }}>{fmt(p.precision)}<span style={{ color: 'var(--ink-4)', fontSize: '0.7rem' }}>%</span></div>
        <div style={{ textAlign: 'right', color: 'var(--ink)' }}>{fmt(p.recall)}<span style={{ color: 'var(--ink-4)', fontSize: '0.7rem' }}>%</span></div>
        <div style={{ textAlign: 'right', color: 'var(--accent)', fontWeight: 600 }}>{fmt(p.f1)}<span style={{ color: 'var(--ink-4)', fontSize: '0.7rem', fontWeight: 400 }}>%</span></div>
        <div style={{ textAlign: 'right', color: 'var(--ink-3)' }}>{p.support}</div>
        <div style={{ textAlign: 'right', color: 'var(--ink-4)', transform: open ? 'rotate(180deg)' : 'none' }}>▾</div>
      </div>
      {open && (
        <div style={{ padding: '1rem 1.1rem 1.1rem', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.5rem' }}>
          {[
            { k: 'TP', full: 'True Positives',  v: p.tp, color: '#1e3a8a',     desc: `predicted ${m.label} & was ${m.label}` },
            { k: 'FP', full: 'False Positives', v: p.fp, color: '#b8431f',     desc: `predicted ${m.label} & was NOT ${m.label}` },
            { k: 'FN', full: 'False Negatives', v: p.fn, color: '#a16207',     desc: `was ${m.label} & predicted something else` },
            { k: 'TN', full: 'True Negatives',  v: p.tn, color: 'var(--ink-3)', desc: `NOT ${m.label} & predicted something else` },
          ].map(box => (
            <div key={box.k} style={{ padding: '0.75rem 0.85rem', background: 'var(--surface)', border: '1px solid var(--rule-soft)', borderTop: `2px solid ${box.color}`, borderRadius: '2px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.66rem', fontWeight: 600, color: box.color, letterSpacing: '0.1em' }}>{box.k}</span>
                <span style={{ fontSize: '0.6rem', color: 'var(--ink-3)', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.05em', textTransform: 'uppercase' }}>{box.full}</span>
              </div>
              <p style={{ fontFamily: "'Fraunces', serif", fontVariationSettings: '"opsz" 144', fontSize: '1.5rem', color: 'var(--ink)', marginTop: '0.3rem' }}>{box.v}</p>
              <p style={{ fontSize: '0.66rem', color: 'var(--ink-3)', marginTop: '0.4rem', lineHeight: 1.45, fontFamily: "'Fraunces', serif", fontStyle: 'italic' }}>{box.desc}</p>
            </div>
          ))}
          <div style={{ gridColumn: '1 / -1', marginTop: '0.4rem', padding: '0.85rem 1rem', background: 'var(--surface)', border: '1px solid var(--rule-soft)', borderRadius: '2px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.74rem' }}>
              {[
                { k: 'Precision', f: 'TP / (TP + FP)',  n: p.tp, d: p.tp + p.fp, v: p.precision, c: 'var(--ink)' },
                { k: 'Recall',    f: 'TP / (TP + FN)',  n: p.tp, d: p.tp + p.fn, v: p.recall,    c: 'var(--ink)' },
                { k: 'F1',        f: '2·P·R / (P + R)', n: null, d: null,        v: p.f1,        c: 'var(--accent)' },
              ].map(row => (
                <div key={row.k}>
                  <p style={{ fontSize: '0.6rem', color: 'var(--ink-3)', letterSpacing: '0.16em', textTransform: 'uppercase', marginBottom: '0.4rem' }}>{row.k}</p>
                  <p style={{ color: 'var(--ink-3)', marginBottom: '0.25rem' }}>{row.f}</p>
                  {row.n !== null && <p style={{ color: 'var(--ink-4)' }}>= {row.n} / {row.d}</p>}
                  {row.k === 'F1' && <p style={{ color: 'var(--ink-4)' }}>= 2·{p.precision.toFixed(3)}·{p.recall.toFixed(3)} / {(p.precision+p.recall).toFixed(3)}</p>}
                  <p style={{ color: row.c, fontWeight: 600, marginTop: '0.3rem' }}>= {fmt(row.v)}%</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── ConfusionMatrix — conventional dark→light blue heatmap (Blues palette) ─
function ConfusionMatrix({ cm, classKeys, maxVal }) {
  const n = cm.length;
  const cellSize = 92;
  // Blues palette: dark navy at high counts → near-white at zero
  const bluesScale = (intensity) => {
    // 0 → very light blue, 1 → deep navy
    const stops = [
      { t: 0.00, c: [247, 251, 255] }, // #f7fbff
      { t: 0.20, c: [222, 235, 247] }, // #deebf7
      { t: 0.40, c: [158, 202, 225] }, // #9ecae1
      { t: 0.60, c: [ 66, 146, 198] }, // #4292c6
      { t: 0.80, c: [ 33,  99, 173] }, // #2163ad
      { t: 1.00, c: [  8,  48, 107] }, // #08306b
    ];
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (intensity >= stops[i].t && intensity <= stops[i+1].t) { lo = stops[i]; hi = stops[i+1]; break; }
    }
    const r = (intensity - lo.t) / (hi.t - lo.t || 1);
    const c = lo.c.map((v, i) => Math.round(v + (hi.c[i] - v) * r));
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  };
  return (
    <div>
      <p style={{ fontSize: '0.78rem', color: 'var(--ink-3)', marginBottom: '1.25rem', fontFamily: "'Fraunces', serif", fontStyle: 'italic' }}>
        Rows: <span style={{ color: '#08306b' }}>true class</span> · Columns: <span style={{ color: '#08306b' }}>predicted class</span>. Color intensity encodes count (Blues colormap).
      </p>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <div style={{ display: 'inline-flex', flexDirection: 'column' }}>
          <div style={{ textAlign: 'center', fontSize: '0.62rem', fontFamily: "'JetBrains Mono', monospace", color: 'var(--ink-2)', letterSpacing: '0.18em', textTransform: 'uppercase', marginBottom: '0.6rem', marginLeft: '160px' }}>Predicted ↓</div>
          <div style={{ display: 'grid', gridTemplateColumns: `160px repeat(${n}, ${cellSize}px)`, gap: '2px' }}>
            <div></div>
            {classKeys.map(k => {
              const meta = CLASS_META[k];
              return (
                <div key={k} style={{ padding: '0.4rem 0.2rem', textAlign: 'center', fontSize: '0.7rem', fontFamily: "'JetBrains Mono', monospace", color: 'var(--ink-2)', fontWeight: 500, lineHeight: 1.25, borderBottom: `2px solid ${meta.color}`, paddingBottom: '0.5rem' }}>
                  {meta.label.split(' ').map((w,i) => <div key={i}>{w}</div>)}
                </div>
              );
            })}
          </div>
          <div style={{ position: 'relative', display: 'flex', marginTop: '0.4rem' }}>
            <div style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', fontSize: '0.62rem', fontFamily: "'JetBrains Mono', monospace", color: 'var(--ink-2)', letterSpacing: '0.18em', textTransform: 'uppercase', display: 'flex', alignItems: 'center', justifyContent: 'center', paddingRight: '0.6rem' }}>True →</div>
            <div style={{ display: 'grid', gridTemplateColumns: `160px repeat(${n}, ${cellSize}px)`, gap: '2px', flex: 1 }}>
              {cm.map((row, i) => (
                <React.Fragment key={i}>
                  <div style={{ padding: '0 0.85rem', fontSize: '0.85rem', fontFamily: "'Fraunces', serif", color: 'var(--ink)', textAlign: 'right', alignSelf: 'center', borderRight: `2px solid ${CLASS_META[classKeys[i]].color}`, paddingRight: '0.85rem' }}>
                    {CLASS_META[classKeys[i]].label}
                  </div>
                  {row.map((val, j) => {
                    const intensity = val / maxVal;
                    const bg = bluesScale(intensity);
                    const isDiag = i === j;
                    const textColor = intensity > 0.55 ? '#f7fbff' : '#08306b';
                    const rowSum = row.reduce((a,b)=>a+b,0);
                    const pct = rowSum ? ((val/rowSum)*100).toFixed(1) : '0';
                    return (
                      <div key={j} style={{ background: bg, padding: '0.85rem 0.3rem', textAlign: 'center', border: isDiag ? '2px solid #08306b' : '1px solid rgba(8,48,107,0.15)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '70px' }}>
                        <span style={{ fontFamily: "'Fraunces', serif", fontVariationSettings: '"opsz" 144', fontSize: '1.3rem', color: textColor, lineHeight: 1 }}>{val}</span>
                        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.62rem', color: textColor, opacity: 0.8, marginTop: '0.3rem' }}>{pct}%</span>
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </div>
      {/* Color scale legend */}
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.6rem', marginTop: '1.75rem' }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', color: 'var(--ink-3)', letterSpacing: '0.05em' }}>0</span>
        <div style={{ display: 'flex', height: 14, border: '1px solid rgba(8,48,107,0.2)' }}>
          {Array.from({ length: 24 }).map((_, i) => (
            <div key={i} style={{ width: 12, height: '100%', background: bluesScale(i / 23) }}></div>
          ))}
        </div>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.65rem', color: 'var(--ink-3)' }}>{maxVal}</span>
        <span style={{ fontFamily: "'Fraunces', serif", fontStyle: 'italic', fontSize: '0.78rem', color: 'var(--ink-3)', marginLeft: '0.4rem' }}>samples</span>
      </div>
    </div>
  );
}

Object.assign(window, { MetricCard, ExpandableMetricRow, ConfusionMatrix });
