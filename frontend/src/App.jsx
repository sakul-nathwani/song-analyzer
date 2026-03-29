import { useState, useRef, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import "./App.css";

// ── Upload box ─────────────────────────────────────────────────────────────

function AudioUploadBox({ label, sublabel, file, onFileChange, color, icon }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragging(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) onFileChange(dropped);
    },
    [onFileChange]
  );

  const handleDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const handleDragLeave = () => setDragging(false);
  const handleClick     = () => inputRef.current?.click();

  return (
    <div
      className={`upload-box ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
      style={{ "--accent": color }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.flac,.aiff,.aif,audio/*"
        style={{ display: "none" }}
        onChange={(e) => e.target.files[0] && onFileChange(e.target.files[0])}
      />
      <div className="upload-icon">{icon}</div>
      <div className="upload-label">{label}</div>
      <div className="upload-sublabel">{sublabel}</div>
      {file ? (
        <div className="upload-filename">
          <span className="file-dot" />
          {file.name}
          <span className="file-size">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
        </div>
      ) : (
        <div className="upload-hint">
          Drop an audio file here or click to browse
          <br />
          <span className="upload-formats">MP3 · WAV · FLAC · AIFF · max 100 MB</span>
        </div>
      )}
    </div>
  );
}

// ── Stat card ──────────────────────────────────────────────────────────────

function StatCard({ label, value, unit, sub }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {value}
        {unit && <span className="stat-unit"> {unit}</span>}
      </div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

// ── Freq bar (overall comparison) ──────────────────────────────────────────

function FreqBar({ label, refPct, wipPct }) {
  return (
    <div className="freq-bar-row">
      <div className="freq-bar-label">{label}</div>
      <div className="freq-bars">
        <div className="freq-bar-track">
          <div className="freq-bar ref-bar" style={{ width: `${refPct}%` }} title={`Ref: ${refPct}%`} />
        </div>
        <div className="freq-bar-track">
          <div className="freq-bar wip-bar" style={{ width: `${wipPct}%` }} title={`WIP: ${wipPct}%`} />
        </div>
      </div>
      <div className="freq-bar-values">
        <span className="ref-text">{refPct}%</span>
        <span className="wip-text">{wipPct}%</span>
      </div>
    </div>
  );
}

// ── Analysis panel ─────────────────────────────────────────────────────────

function AnalysisPanel({ analysis, label, color }) {
  if (!analysis) return null;
  return (
    <div className="analysis-panel" style={{ "--accent": color }}>
      <div className="panel-header">
        <span className="panel-dot" />
        {label}
      </div>
      <div className="stats-grid">
        <StatCard label="Duration"        value={analysis.duration_seconds} unit="s" />
        <StatCard label="Tempo"           value={analysis.tempo_bpm}        unit="BPM" />
        <StatCard label="Key"             value={analysis.key} />
        <StatCard label="Avg Loudness"    value={analysis.loudness.average_db}       unit="dBFS" />
        <StatCard label="Peak Loudness"   value={analysis.loudness.peak_db}          unit="dBFS" />
        <StatCard label="Dynamic Range"   value={analysis.loudness.dynamic_range_db} unit="dB" />
        <StatCard label="Spectral Centroid" value={Math.round(analysis.frequency_spectrum.spectral_centroid_hz)} unit="Hz" sub="brightness" />
        <StatCard label="Harmonic Ratio"  value={analysis.harmonic_ratio} sub="tonal richness" />
        <StatCard label="Sections"        value={analysis.sections.length} sub="detected" />
      </div>
    </div>
  );
}

// ── Progress steps ─────────────────────────────────────────────────────────

const STAGE_LABELS = {
  idle: null, uploading: "Uploading files...", extracting: "Extracting audio features...",
  separating: "Separating stems via Demucs...",
  generating: "Generating AI feedback...",
  done: null, error: null,
};

function ProgressSteps({ stage, deepAnalysis }) {
  const steps = [
    { key: "uploading",  label: "Upload" },
    { key: "extracting", label: "Analyze Audio" },
    ...(deepAnalysis ? [{ key: "separating", label: "Separate Stems" }] : []),
    { key: "generating", label: "AI Feedback" },
  ];
  const order = [
    "uploading", "extracting",
    ...(deepAnalysis ? ["separating"] : []),
    "generating", "done",
  ];
  const current = order.indexOf(stage);
  return (
    <div className="progress-steps">
      {steps.map((step, i) => {
        const done   = current > i || stage === "done";
        const active = current === i && stage !== "done";
        return (
          <div key={step.key} className={`step ${done ? "done" : ""} ${active ? "active" : ""}`}>
            <div className="step-dot">
              {done ? "✓" : active ? <span className="step-spinner" /> : i + 1}
            </div>
            <span className="step-label">{step.label}</span>
            {i < steps.length - 1 && <div className={`step-line ${done ? "done" : ""}`} />}
          </div>
        );
      })}
    </div>
  );
}

// ── Priority scores panel ──────────────────────────────────────────────────

function PriorityScoresPanel({ scores }) {
  if (!scores || scores.length === 0) return null;
  const scoreColor = (s) => s >= 8 ? "#ff6584" : s >= 5 ? "#ffb347" : "#43d9ad";
  return (
    <section className="priority-section">
      <h2 className="section-title">Priority Issues</h2>
      <div className="priority-list">
        {scores.slice(0, 3).map((item, i) => (
          <div key={i} className="priority-item">
            <div className="priority-badge" style={{ background: scoreColor(item.score) }}>
              {item.score}
            </div>
            <div className="priority-content">
              <div className="priority-label">{item.label}</div>
              <div className="priority-summary">{item.summary}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ── Section-by-section comparison ─────────────────────────────────────────

const FREQ_KEYS   = ["sub_bass_pct", "bass_pct", "low_mids_pct", "mids_pct", "high_mids_pct", "highs_pct"];
const FREQ_LABELS = ["Sub", "Bass", "Lo-M", "Mids", "Hi-M", "Highs"];

// EDM section types in canonical display order.
// Un-numbered fallbacks (Verse, Buildup, Drop) appear first so they show up
// if the AI returns labels without a number suffix.
const SECTION_ORDER = [
  "Intro",
  "Verse",  "Verse 1",  "Buildup",  "Buildup 1",  "Drop",  "Drop 1",
  "Breakdown",
  "Verse 2", "Buildup 2", "Drop 2",
  "Breakdown 2",
  "Verse 3", "Buildup 3", "Drop 3",
  "Outro",
];

function groupByLabel(sections) {
  const groups = {};
  for (const sec of (sections || [])) {
    const l = sec.label;
    if (!groups[l]) groups[l] = [];
    groups[l].push(sec);
  }
  return groups;
}

function avgSections(secs) {
  if (!secs || secs.length === 0) return null;
  const n = secs.length;
  const loudness = Math.round(
    secs.reduce((s, sec) => s + (sec.avg_loudness_db || 0), 0) / n * 10
  ) / 10;
  const freq = {};
  for (const key of FREQ_KEYS) {
    freq[key] = Math.round(
      secs.reduce((s, sec) => s + (sec.frequency_balance?.[key] || 0), 0) / n * 10
    ) / 10;
  }
  return { loudness, freq, count: n };
}

function SectionComparisonPanel({ refAnalysis, wipAnalysis }) {
  if (!refAnalysis || !wipAnalysis) return null;

  const refGroups = groupByLabel(refAnalysis.sections);
  const wipGroups = groupByLabel(wipAnalysis.sections);

  // All label types present in either track, in canonical EDM order
  const allLabels = SECTION_ORDER.filter((l) => refGroups[l] || wipGroups[l]);
  if (allLabels.length === 0) return null;

  return (
    <div className="section-comparison">
      <div className="freq-header">
        <span className="freq-title">Section Type Comparison</span>
        <div className="freq-legend">
          <span className="legend-dot ref-dot" /> Reference
          <span className="legend-dot wip-dot" /> WIP
        </div>
      </div>
      <div className="section-cards">
        {allLabels.map((label) => {
          const refAvg = avgSections(refGroups[label]);
          const wipAvg = avgSections(wipGroups[label]);
          const countMismatch = (refAvg?.count ?? 0) !== (wipAvg?.count ?? 0);

          return (
            <div key={label} className="section-card">
              <div className="section-type-header">
                <span className="section-type-name">{label}</span>
                <div className="section-type-counts">
                  <span className="ref-text">Ref ×{refAvg?.count ?? 0}</span>
                  <span className="wip-text">WIP ×{wipAvg?.count ?? 0}</span>
                  {countMismatch && (
                    <span className="count-mismatch" title="Count differs between tracks">⚠</span>
                  )}
                </div>
              </div>

              {!refAvg && <div className="section-absent wip-only">Missing in reference</div>}
              {!wipAvg && <div className="section-absent wip-missing">Missing in WIP</div>}

              {(refAvg || wipAvg) && (
                <>
                  <div className="section-loudness-row">
                    <span className="section-stat-lbl">Avg Loudness</span>
                    <span className="ref-text">{refAvg ? `${refAvg.loudness} dB` : "—"}</span>
                    <span className="wip-text">{wipAvg ? `${wipAvg.loudness} dB` : "—"}</span>
                  </div>
                  <div className="section-freq-grid">
                    {FREQ_KEYS.map((key, ki) => {
                      const rv = refAvg?.freq?.[key] ?? 0;
                      const wv = wipAvg?.freq?.[key] ?? 0;
                      return (
                        <div key={key} className="section-freq-row">
                          <span className="section-freq-lbl">{FREQ_LABELS[ki]}</span>
                          <div className="freq-bars">
                            <div className="freq-bar-track">
                              <div className="freq-bar ref-bar" style={{ width: `${rv}%` }} />
                            </div>
                            <div className="freq-bar-track">
                              <div className="freq-bar wip-bar" style={{ width: `${wv}%` }} />
                            </div>
                          </div>
                          <div className="freq-bar-values" style={{ minWidth: 70 }}>
                            <span className="ref-text">{rv}%</span>
                            <span className="wip-text">{wv}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function sectionChipColor(label) {
  if (label.startsWith("Drop"))      return "#ff6584";
  if (label.startsWith("Buildup"))   return "#ffb347";
  if (label.startsWith("Verse"))     return "#6c63ff";
  if (label === "Intro" || label === "Outro") return "#4ecdc4";
  if (label.startsWith("Breakdown")) return "#43d9ad";
  return "#aaaacc";
}

// ── Waveform chart (SVG energy envelope) ──────────────────────────────────

function WaveformChart({ refAnalysis, wipAnalysis }) {
  if (!refAnalysis?.energy_profile?.length || !wipAnalysis?.energy_profile?.length) return null;

  const W = 860, H = 150;
  const PL = 40, PR = 12, PT = 12, PB = 28;
  const PW = W - PL - PR;
  const PH = H - PT - PB;

  const refP = refAnalysis.energy_profile;
  const wipP = wipAnalysis.energy_profile;

  // Use actual audio duration so section markers near the end aren't clipped
  const maxTime = Math.max(
    refAnalysis.duration_seconds ?? 0,
    wipAnalysis.duration_seconds ?? 0,
    refP.at(-1)?.time ?? 0,
    wipP.at(-1)?.time ?? 0,
    1
  );
  const maxEnergy = Math.max(...refP.map((p) => p.energy), ...wipP.map((p) => p.energy), 1e-9);

  const tx = (t) => (PL + (t / maxTime) * PW).toFixed(1);
  const ty = (e) => (PT + PH - (e / maxEnergy) * PH).toFixed(1);

  const linePath = (pts) =>
    pts.length === 0 ? "" : "M " + pts.map((p) => `${tx(p.time)},${ty(p.energy)}`).join(" L ");

  const fillPath = (pts) => {
    if (!pts.length) return "";
    const y0 = (PT + PH).toFixed(1);
    const line = pts.map((p) => `${tx(p.time)},${ty(p.energy)}`).join(" L ");
    return `M ${tx(pts[0].time)},${y0} L ${line} L ${tx(pts.at(-1).time)},${y0} Z`;
  };

  const tickInterval = maxTime > 180 ? 60 : 30;
  const xTicks = [];
  for (let t = 0; t <= maxTime; t += tickInterval) {
    const m = Math.floor(t / 60);
    const s = t % 60;
    xTicks.push({ x: tx(t), label: m > 0 ? `${m}:${String(s).padStart(2, "0")}` : `${t}s` });
  }

  return (
    <div className="waveform-section">
      <div className="freq-header">
        <span className="freq-title">Energy Envelope</span>
        <div className="freq-legend">
          <span className="legend-dot ref-dot" /> Reference
          <span className="legend-dot wip-dot" /> WIP
        </div>
      </div>
      <div className="waveform-chart-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} className="waveform-svg" preserveAspectRatio="none">
          {/* Y grid */}
          {[0.25, 0.5, 0.75, 1.0].map((pct) => {
            const y = ty(maxEnergy * pct);
            return (
              <g key={pct}>
                <line x1={PL} y1={y} x2={W - PR} y2={y} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                <text x={PL - 5} y={Number(y) + 4} textAnchor="end" fill="rgba(255,255,255,0.25)" fontSize="9">
                  {Math.round(pct * 100)}%
                </text>
              </g>
            );
          })}

          {/* Fills */}
          <path d={fillPath(refP)} fill="rgba(108,99,255,0.07)" />
          <path d={fillPath(wipP)} fill="rgba(255,101,132,0.07)" />

          {/* Section markers — Ref (purple, lines only — drawn first so WIP labels sit on top) */}
          {(refAnalysis.sections ?? []).slice(1).map((sec, i) => (
            <line key={`rs${i}`} x1={tx(sec.start)} y1={PT} x2={tx(sec.start)} y2={PT + PH}
              stroke="rgba(108,99,255,0.2)" strokeWidth="1" strokeDasharray="3,3" />
          ))}

          {/* Section markers — WIP (pink lines + labels for every section) */}
          {(() => {
            const secs = wipAnalysis.sections ?? [];
            let lastLabelX = -999;
            let rowIndex = 0;
            const yRows = [PT + 10, PT + 21, PT + 32];
            return secs.map((sec, i) => {
              const xVal = i === 0 ? PL + 3 : Number(tx(sec.start)) + 3;
              const showLabel = xVal - lastLabelX >= 28;
              if (showLabel) {
                lastLabelX = xVal;
                rowIndex = (rowIndex + 1) % yRows.length;
              }
              return (
                <g key={`ws${i}`}>
                  {i > 0 && (
                    <line x1={tx(sec.start)} y1={PT} x2={tx(sec.start)} y2={PT + PH}
                      stroke="rgba(255,101,132,0.35)" strokeWidth="1" strokeDasharray="3,3" />
                  )}
                  {showLabel && (
                    <text
                      x={xVal}
                      y={yRows[rowIndex]}
                      fill="rgba(255,101,132,0.7)"
                      fontSize="8.5"
                    >
                      {sec.label}
                    </text>
                  )}
                </g>
              );
            });
          })()}

          {/* Lines */}
          <path d={linePath(refP)} fill="none" stroke="#6c63ff" strokeWidth="1.5" />
          <path d={linePath(wipP)} fill="none" stroke="#ff6584" strokeWidth="1.5" />

          {/* Axes */}
          <line x1={PL} y1={PT} x2={PL} y2={PT + PH} stroke="rgba(255,255,255,0.12)" strokeWidth="1" />
          <line x1={PL} y1={PT + PH} x2={W - PR} y2={PT + PH} stroke="rgba(255,255,255,0.12)" strokeWidth="1" />

          {/* X ticks */}
          {xTicks.map((tk, i) => (
            <g key={i}>
              <line x1={tk.x} y1={PT + PH} x2={tk.x} y2={PT + PH + 4}
                stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
              <text x={tk.x} y={PT + PH + 14} textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="9">
                {tk.label}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}

// ── Sidechain comparison panel ─────────────────────────────────────────────

function SidechainPanel({ refAnalysis, wipAnalysis }) {
  const refSC = refAnalysis?.sidechain;
  const wipSC = wipAnalysis?.sidechain;
  if (!refSC && !wipSC) return null;

  function SCTrack({ sc, color }) {
    if (!sc) return <span style={{ color: "var(--text-muted)" }}>—</span>;
    return (
      <div className="sidechain-track">
        <div className={`sidechain-detected ${sc.detected ? "yes" : "no"}`}>
          {sc.detected ? "Detected" : "Not Detected"}
        </div>
        {sc.detected ? (
          <div className="sidechain-stats">
            <div className="sc-stat">
              <span className="sc-stat-lbl">Depth</span>
              <span className="sc-stat-val" style={{ color }}>{sc.depth_db} dB</span>
            </div>
            {sc.release_ms != null && (
              <div className="sc-stat">
                <span className="sc-stat-lbl">Release</span>
                <span className="sc-stat-val" style={{ color }}>{sc.release_ms} ms</span>
              </div>
            )}
            {sc.rate && (
              <div className="sc-stat">
                <span className="sc-stat-lbl">Rate</span>
                <span className="sc-stat-val" style={{ color }}>{sc.rate}</span>
              </div>
            )}
            <div className="sc-stat">
              <span className="sc-stat-lbl">Consistency</span>
              <span className="sc-stat-val" style={{ color }}>{Math.round((sc.consistency ?? 0) * 100)}%</span>
            </div>
          </div>
        ) : (
          sc.depth_db > 0 && (
            <div className="sidechain-hint">Avg depth: {sc.depth_db} dB (below detection threshold)</div>
          )
        )}
      </div>
    );
  }

  // Determine flag to show
  let flag = null;
  const refDet = refSC?.detected ?? false;
  const wipDet = wipSC?.detected ?? false;
  if (refDet && !wipDet) {
    flag = { type: "warning", msg: "Reference has sidechain compression but your WIP does not. Consider adding sidechain to bass/pad elements." };
  } else if (refDet && wipDet) {
    const diff = (wipSC?.depth_db ?? 0) - (refSC?.depth_db ?? 0);
    if (diff < -3)
      flag = { type: "warning", msg: `WIP sidechain is ${Math.abs(diff).toFixed(1)} dB more subtle than the reference. Increase depth for more pump.` };
    else if (diff > 3)
      flag = { type: "info", msg: `WIP sidechain is ${diff.toFixed(1)} dB more aggressive than the reference. Consider reducing for a cleaner mix.` };
  }

  return (
    <div className="sidechain-panel">
      <div className="freq-header" style={{ marginBottom: 14 }}>
        <span className="freq-title">Sidechain Compression</span>
      </div>
      <div className="sidechain-grid">
        <div className="sidechain-col">
          <div className="sidechain-col-header ref-text">Reference</div>
          <SCTrack sc={refSC} color="var(--purple)" />
        </div>
        <div className="sidechain-col">
          <div className="sidechain-col-header wip-text">WIP</div>
          <SCTrack sc={wipSC} color="var(--pink)" />
        </div>
      </div>
      {flag && (
        <div className={`sidechain-flag ${flag.type}`}>
          {flag.type === "warning" ? "⚠ " : "ℹ "}{flag.msg}
        </div>
      )}
    </div>
  );
}

// ── Chat panel ─────────────────────────────────────────────────────────────

function ChatPanel({ jobId }) {
  const [messages, setMessages] = useState([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async () => {
    if (!input.trim() || loading || !jobId) return;
    const userMsg    = { role: "user", content: input.trim() };
    const nextMsgs   = [...messages, userMsg];
    setMessages(nextMsgs);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`/chat/${jobId}`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ messages: nextMsgs }),
      });
      if (!res.ok) {
        let err = `Error ${res.status}`;
        try { err = (await res.json()).detail || err; } catch {}
        throw new Error(err);
      }
      const data = await res.json();
      setMessages([...nextMsgs, { role: "assistant", content: data.reply }]);
    } catch (err) {
      setMessages([...nextMsgs, { role: "assistant", content: `Sorry, something went wrong: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <section className="chat-section">
      <h2 className="section-title">Ask a Follow-up Question</h2>
      <div className="chat-card">
        {messages.length === 0 && (
          <div className="chat-placeholder">
            Ask anything about your tracks — e.g. <em>"Which plugin should I use to fix the sub bass?"</em> or <em>"What's causing the brightness issue?"</em>
          </div>
        )}
        <div className="chat-messages">
          {messages.map((m, i) => (
            <div key={i} className={`chat-message ${m.role}`}>
              <div className="chat-bubble">
                {m.role === "assistant"
                  ? <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]} components={MD_COMPONENTS}>{m.content}</ReactMarkdown></div>
                  : m.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="chat-message assistant">
              <div className="chat-bubble chat-loading">
                <span className="step-spinner" /> Thinking...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        <div className="chat-input-row">
          <textarea
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a question about your mix..."
            rows={2}
            disabled={loading}
          />
          <button
            className={`chat-send-btn ${input.trim() && !loading ? "active" : ""}`}
            onClick={send}
            disabled={!input.trim() || loading}
          >
            Send
          </button>
        </div>
      </div>
    </section>
  );
}

// ── History panel ──────────────────────────────────────────────────────────

function HistoryPanel({ history, onClear, onClose }) {
  const scoreColor = (s) => s >= 8 ? "#ff6584" : s >= 5 ? "#ffb347" : "#43d9ad";
  return (
    <div className="history-overlay" onClick={onClose}>
      <div className="history-panel" onClick={(e) => e.stopPropagation()}>
        <div className="history-header">
          <span>Analysis History</span>
          <div style={{ display: "flex", gap: 8 }}>
            {history.length > 0 && (
              <button className="history-clear-btn" onClick={onClear}>Clear</button>
            )}
            <button className="history-close-btn" onClick={onClose}>✕</button>
          </div>
        </div>
        {history.length === 0 ? (
          <div className="history-empty">No analyses saved yet. Run your first analysis to start tracking progress.</div>
        ) : (
          <div className="history-list">
            {history.map((entry) => (
              <div key={entry.id} className="history-item">
                <div className="history-item-top">
                  <span className="history-wip">{entry.wip_name}</span>
                  <span className="history-date">
                    {new Date(entry.timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
                <div className="history-refs">vs {entry.ref_names.join(", ")}</div>
                <div className="history-scores">
                  {(entry.priority_scores || []).slice(0, 3).map((s, i) => (
                    <div key={i} className="history-score-row">
                      <span className="history-score-badge" style={{ background: scoreColor(s.score) }}>{s.score}</span>
                      <span className="history-score-label">{s.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Stem analysis panel ────────────────────────────────────────────────────

const STEM_LABELS = { drums: "Drums", bass: "Bass", other: "Other" };
const STEM_ORDER  = ["drums", "bass", "other"];

function StemFreqMini({ fb, color }) {
  if (!fb) return null;
  const entries = [
    ["Sub",  fb.sub_bass_pct],
    ["Bass", fb.bass_pct],
    ["LM",   fb.low_mids_pct],
    ["M",    fb.mids_pct],
    ["HM",   fb.high_mids_pct],
    ["Hi",   fb.highs_pct],
  ];
  return (
    <div className="stem-freq-mini">
      {entries.map(([lbl, pct]) => (
        <div key={lbl} className="stem-freq-mini-row">
          <span className="stem-freq-mini-lbl">{lbl}</span>
          <div className="stem-freq-mini-track">
            <div className="stem-freq-mini-bar" style={{ width: `${pct}%`, background: color }} />
          </div>
          <span className="stem-freq-mini-val">{pct}%</span>
        </div>
      ))}
    </div>
  );
}

function StemAnalysisPanel({ stemAnalyses }) {
  if (!stemAnalyses) return null;
  const refStems = stemAnalyses.reference || {};
  const wipStems = stemAnalyses.wip       || {};
  const hasData  = STEM_ORDER.some((s) => refStems[s] || wipStems[s]);
  if (!hasData) return null;

  return (
    <section className="stem-section">
      <h2 className="section-title">Stem Analysis <span className="deep-badge">Deep</span></h2>
      <div className="stem-grid">
        {STEM_ORDER.map((stemKey) => {
          const refS = refStems[stemKey];
          const wipS = wipStems[stemKey];
          if (!refS && !wipS) return null;
          return (
            <div key={stemKey} className="stem-card">
              <div className="stem-card-header">{STEM_LABELS[stemKey]}</div>
              <div className="stem-card-cols">
                {[{ label: "Reference", data: refS, color: "#6c63ff" },
                  { label: "WIP",       data: wipS, color: "#ff6584" }].map(({ label, data, color }) => (
                  <div key={label} className="stem-card-col">
                    <div className="stem-col-label" style={{ color }}>{label}</div>
                    {data ? (
                      <>
                        <div className="stem-stat-row">
                          <span className="stem-stat-lbl">Avg</span>
                          <span className="stem-stat-val" style={{ color }}>{data.loudness.average_db} dBFS</span>
                        </div>
                        <div className="stem-stat-row">
                          <span className="stem-stat-lbl">Peak</span>
                          <span className="stem-stat-val" style={{ color }}>{data.loudness.peak_db} dBFS</span>
                        </div>
                        <div className="stem-stat-row">
                          <span className="stem-stat-lbl">DR</span>
                          <span className="stem-stat-val" style={{ color }}>{data.loudness.dynamic_range_db} dB</span>
                        </div>
                        <div className="stem-stat-row">
                          <span className="stem-stat-lbl">Centroid</span>
                          <span className="stem-stat-val" style={{ color }}>{Math.round(data.spectral_centroid_hz)} Hz</span>
                        </div>
                        <StemFreqMini fb={data.frequency_balance} color={color} />
                      </>
                    ) : (
                      <span className="stem-absent">—</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ── Main app ───────────────────────────────────────────────────────────────

// ── Markdown table renderer ────────────────────────────────────────────────
// Detects delta values (e.g. "+5 dB", "−3%", "-2.1 kHz") in cells and
// applies green/red colouring so differences stand out in the EQ table.

function deltaClass(text) {
  const s = String(text).trim();
  if (/^[+＋]/.test(s) || /^\+\d/.test(s)) return "td-positive";
  if (/^[-−–]/.test(s) || /^−\d/.test(s))  return "td-negative";
  return "";
}

const MD_COMPONENTS = {
  table: (props) => (
    <div className="md-table-wrap">
      <table className="md-table" {...props} />
    </div>
  ),
  thead: (props) => <thead className="md-thead" {...props} />,
  tbody: (props) => <tbody {...props} />,
  tr:    (props) => <tr className="md-tr" {...props} />,
  th:    ({ children, style, ...props }) => (
    <th className="md-th" style={style} {...props}>{children}</th>
  ),
  td:    ({ children, style, ...props }) => {
    const text = typeof children === "string" ? children
      : Array.isArray(children) ? children.join("") : "";
    return (
      <td className={`md-td ${deltaClass(text)}`} style={style} {...props}>
        {children}
      </td>
    );
  },
};

const freqLabels = [
  ["Sub-bass",  "sub_bass_pct"],
  ["Bass",      "bass_pct"],
  ["Low-mids",  "low_mids_pct"],
  ["Mids",      "mids_pct"],
  ["High-mids", "high_mids_pct"],
  ["Highs",     "highs_pct"],
];

export default function App() {
  const [refFiles,       setRefFiles]       = useState([null]);
  const [wipFile,        setWipFile]        = useState(null);
  const [stage,          setStageState]     = useState("idle");
  const [suggestions,    setSuggestions]    = useState("");
  const [refAnalysis,    setRefAnalysis]    = useState(null);
  const [wipAnalysis,    setWipAnalysis]    = useState(null);
  const [priorityScores, setPriorityScores] = useState([]);
  const [stemAnalyses,   setStemAnalyses]   = useState(null);
  const [stemError,      setStemError]      = useState(null);
  const [activeTab,      setActiveTab]      = useState("overview");
  const [deepAnalysis,   setDeepAnalysis]   = useState(false);
  const [jobId,          setJobId]          = useState(null);
  const [error,          setError]          = useState("");
  const [elapsed,        setElapsed]        = useState(0);
  const [showHistory,    setShowHistory]    = useState(false);
  const [history,        setHistory]        = useState(() => {
    try { return JSON.parse(localStorage.getItem("songAnalyzerHistory") || "[]"); }
    catch { return []; }
  });

  const timerRef = useRef(null);
  const pollRef  = useRef(null);
  const stageRef = useRef("idle");

  const setStage = (s) => { stageRef.current = s; setStageState(s); };

  const isAnalyzing = stage === "uploading" || stage === "extracting" || stage === "separating" || stage === "generating";

  useEffect(() => {
    if (isAnalyzing) {
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isAnalyzing]);

  useEffect(() => { return () => clearInterval(pollRef.current); }, []);

  // Multiple ref helpers
  const addRef    = () => { if (refFiles.length < 3) setRefFiles([...refFiles, null]); };
  const removeRef = (i) => setRefFiles(refFiles.filter((_, idx) => idx !== i));
  const setRefAt  = (i, file) => { const updated = [...refFiles]; updated[i] = file; setRefFiles(updated); };

  const validRefs  = refFiles.filter(Boolean);
  const canAnalyze = validRefs.length > 0 && wipFile && !isAnalyzing;

  const saveToHistory = (entry) => {
    const next = [entry, ...history].slice(0, 20);
    setHistory(next);
    localStorage.setItem("songAnalyzerHistory", JSON.stringify(next));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("songAnalyzerHistory");
  };

  const handleStop = () => {
    clearInterval(pollRef.current);
    if (jobId) {
      fetch(`/cancel/${jobId}`, { method: "POST" }).catch(() => {});
    }
    setStage("idle");
    setJobId(null);
    setSuggestions("");
    setRefAnalysis(null);
    setWipAnalysis(null);
    setPriorityScores([]);
    setStemAnalyses(null);
    setStemError(null);
    setError("");
    setActiveTab("overview");
  };

  const handleAnalyze = async () => {
    if (!canAnalyze) return;

    setStage("uploading");
    setSuggestions("");
    setRefAnalysis(null);
    setWipAnalysis(null);
    setPriorityScores([]);
    setStemAnalyses(null);
    setStemError(null);
    setJobId(null);
    setError("");
    setActiveTab("overview");

    const formData = new FormData();
    validRefs.forEach((f) => formData.append("references", f));
    formData.append("wip", wipFile);
    formData.append("deep_analysis", deepAnalysis ? "true" : "false");

    try {
      const res = await fetch("/analyze", { method: "POST", body: formData });

      if (!res.ok) {
        let msg = `Server error ${res.status}`;
        try { const b = await res.json(); msg = b.detail || msg; }
        catch { try { msg = (await res.text()).slice(0, 200) || msg; } catch {} }
        throw new Error(msg);
      }

      const { job_id } = await res.json();
      setJobId(job_id);
      setStage("extracting");

      clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await fetch(`/status/${job_id}`);
          if (!statusRes.ok) {
            let msg = `Status check failed (${statusRes.status})`;
            try { const b = await statusRes.json(); msg = b.detail || msg; } catch {}
            throw new Error(msg);
          }
          const data = await statusRes.json();

          if (data.stage === "separating" && stageRef.current !== "separating") {
            setStage("separating");
          }
          if (data.stage === "generating" && stageRef.current !== "generating") {
            setStage("generating");
          }

          if (data.status === "done") {
            clearInterval(pollRef.current);
            const scores = data.result.priority_scores || [];
            setRefAnalysis(data.result.reference);
            setWipAnalysis(data.result.wip);
            setSuggestions(data.result.suggestions);
            setPriorityScores(scores);
            setStemAnalyses(data.result.stem_analyses || null);
            setStemError(data.stem_error || data.result?.stem_error || null);
            setStage("done");
            saveToHistory({
              id:              job_id,
              timestamp:       new Date().toISOString(),
              wip_name:        wipFile.name,
              ref_names:       validRefs.map((f) => f.name),
              priority_scores: scores,
              job_id,
            });
          } else if (data.status === "error") {
            clearInterval(pollRef.current);
            throw new Error(data.error || "Analysis failed on server");
          }
        } catch (pollErr) {
          clearInterval(pollRef.current);
          setError(pollErr.message);
          setStage("error");
        }
      }, 3000);
    } catch (err) {
      setError(err.message);
      setStage("error");
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-row">
            <div className="logo">
              <span className="logo-icon">🎧</span>
              <span className="logo-text">Song Analyzer</span>
            </div>
            <button className="history-btn" onClick={() => setShowHistory(true)}>
              History {history.length > 0 && <span className="history-count">{history.length}</span>}
            </button>
          </div>
          <p className="header-subtitle">
            AI-powered music production feedback — compare your WIP to a reference track
          </p>
        </div>
      </header>

      {showHistory && (
        <HistoryPanel
          history={history}
          onClear={clearHistory}
          onClose={() => setShowHistory(false)}
        />
      )}

      <main className="app-main">
        {/* Upload */}
        <section className="upload-section">
          <div className="upload-grid">
            <div className="upload-refs-col">
              {refFiles.map((file, i) => (
                <div key={i} className="upload-ref-row">
                  <AudioUploadBox
                    label={refFiles.length > 1 ? `Reference Track ${i + 1}` : "Reference Track"}
                    sublabel="The song you're aiming for · first 6 min analysed"
                    file={file}
                    onFileChange={(f) => setRefAt(i, f)}
                    color="#6c63ff"
                    icon="🎯"
                  />
                  {refFiles.length > 1 && (
                    <button className="remove-ref-btn" onClick={() => removeRef(i)} title="Remove">✕</button>
                  )}
                </div>
              ))}
              {refFiles.length < 3 && (
                <button className="add-ref-btn" onClick={addRef}>
                  + Add another reference
                </button>
              )}
            </div>

            <div className="upload-vs">VS</div>

            <AudioUploadBox
              label="Your WIP"
              sublabel="Your work-in-progress · first 6 min analysed"
              file={wipFile}
              onFileChange={setWipFile}
              color="#ff6584"
              icon="🎛️"
            />
          </div>

          <div className="analyze-row">
            <button
              className={`analyze-btn ${canAnalyze ? "active" : ""} ${isAnalyzing ? "loading" : ""}`}
              onClick={handleAnalyze}
              disabled={!canAnalyze}
            >
              {isAnalyzing ? (
                <><span className="spinner" />{STAGE_LABELS[stage]}</>
              ) : (
                <><span className="btn-icon">✦</span>Run Analysis</>
              )}
            </button>

            {isAnalyzing && (
              <button className="stop-btn" onClick={handleStop} title="Cancel analysis">
                ✕ Stop
              </button>
            )}

            <label className={`deep-toggle ${isAnalyzing ? "disabled" : ""}`}>
              <input
                type="checkbox"
                checked={deepAnalysis}
                onChange={(e) => !isAnalyzing && setDeepAnalysis(e.target.checked)}
                disabled={isAnalyzing}
              />
              <span className="deep-toggle-track">
                <span className="deep-toggle-thumb" />
              </span>
              <span className="deep-toggle-label">
                Deep Analysis
                <span className="deep-toggle-sub">Stem separation via Demucs · ~$0.009/run</span>
              </span>
            </label>
          </div>

          {isAnalyzing && (
            <div className="analysis-progress">
              <ProgressSteps stage={stage} deepAnalysis={deepAnalysis} />
              {isAnalyzing && (
                <div className="elapsed">
                  {elapsed}s elapsed
                  {stage === "separating" && " — Demucs stem separation running on GPU, this takes 1–3 min"}
                  {stage !== "separating" && " — large files can take 30–60 seconds"}
                </div>
              )}
            </div>
          )}
        </section>


        {/* Error */}
        {stage === "error" && (
          <div className="error-box"><span className="error-icon">⚠️</span> {error}</div>
        )}

        {/* Tabbed results */}
        {stage === "done" && refAnalysis && wipAnalysis && (
          <section className="results-tabs-section">
            <div className="tab-bar">
              {[
                { id: "overview",   label: "Overview"    },
                { id: "frequency",  label: "Frequency"   },
                { id: "stems",      label: "Stems"       },
                { id: "feedback",   label: "AI Feedback" },
              ].map(({ id, label }) => (
                <button
                  key={id}
                  className={`tab-btn${activeTab === id ? " active" : ""}`}
                  onClick={() => setActiveTab(id)}
                >
                  {label}
                </button>
              ))}
            </div>

            <div className="tab-panel">

              {activeTab === "overview" && (
                <>
                  <PriorityScoresPanel scores={priorityScores} />
                  <WaveformChart refAnalysis={refAnalysis} wipAnalysis={wipAnalysis} />
                  <SectionComparisonPanel refAnalysis={refAnalysis} wipAnalysis={wipAnalysis} />
                </>
              )}

              {activeTab === "frequency" && (
                <>
                  <div className="analysis-grid">
                    <AnalysisPanel analysis={refAnalysis} label={validRefs.length > 1 ? `Averaged Target (${validRefs.length} refs)` : "Reference Track"} color="#6c63ff" />
                    <AnalysisPanel analysis={wipAnalysis} label="Your WIP" color="#ff6584" />
                  </div>
                  <div className="freq-comparison">
                    <div className="freq-header">
                      <span className="freq-title">Overall Frequency Balance</span>
                      <div className="freq-legend">
                        <span className="legend-dot ref-dot" /> Reference
                        <span className="legend-dot wip-dot" /> WIP
                      </div>
                    </div>
                    <div className="freq-bars-list">
                      {freqLabels.map(([label, key]) => (
                        <FreqBar
                          key={key}
                          label={label}
                          refPct={refAnalysis.frequency_balance[key]}
                          wipPct={wipAnalysis.frequency_balance[key]}
                        />
                      ))}
                    </div>
                  </div>
                </>
              )}

              {activeTab === "stems" && (
                <>
                  {stemError && (
                    <div className="stem-error-box">
                      <span className="stem-error-icon">ⓘ</span>
                      <div>
                        <strong>Deep Analysis unavailable</strong> — regular analysis completed successfully.
                        <div className="stem-error-detail">{stemError}</div>
                      </div>
                    </div>
                  )}
                  {stemAnalyses ? (
                    <StemAnalysisPanel stemAnalyses={stemAnalyses} />
                  ) : !stemError && (
                    <div className="tab-empty">
                      <span className="tab-empty-icon">🎛️</span>
                      <p>Enable <strong>Deep Analysis</strong> before running to see stem-level breakdown (vocals, drums, bass, other).</p>
                    </div>
                  )}
                </>
              )}

              {activeTab === "feedback" && (
                <>
                  <div className="suggestions-card">
                    <div className="markdown-body">
                      <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]} components={MD_COMPONENTS}>
                        {suggestions}
                      </ReactMarkdown>
                    </div>
                  </div>
                  <ChatPanel jobId={jobId} />
                </>
              )}

            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        Powered by <strong>librosa</strong> + <strong>Claude Opus 4.6</strong>
      </footer>
    </div>
  );
}
