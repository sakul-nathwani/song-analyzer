import { useState, useRef, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
        accept=".mp3,.wav,.flac,.ogg,.aiff,.aif,.m4a,audio/*"
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
          <span className="upload-formats">MP3 · WAV · FLAC · OGG · AIFF · M4A</span>
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
  generating: "Generating AI feedback...", done: null, error: null,
};
const STEPS = [
  { key: "uploading",  label: "Upload" },
  { key: "extracting", label: "Analyze Audio" },
  { key: "generating", label: "AI Feedback" },
];

function ProgressSteps({ stage }) {
  const order   = ["uploading", "extracting", "generating", "done"];
  const current = order.indexOf(stage);
  return (
    <div className="progress-steps">
      {STEPS.map((step, i) => {
        const done   = current > i || stage === "done";
        const active = current === i && stage !== "done";
        return (
          <div key={step.key} className={`step ${done ? "done" : ""} ${active ? "active" : ""}`}>
            <div className="step-dot">
              {done ? "✓" : active ? <span className="step-spinner" /> : i + 1}
            </div>
            <span className="step-label">{step.label}</span>
            {i < STEPS.length - 1 && <div className={`step-line ${done ? "done" : ""}`} />}
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
        {scores.map((item, i) => (
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

function SectionComparisonPanel({ refAnalysis, wipAnalysis }) {
  if (!refAnalysis || !wipAnalysis) return null;
  const refSecs = refAnalysis.sections || [];
  const wipSecs = wipAnalysis.sections || [];
  const maxLen  = Math.max(refSecs.length, wipSecs.length);
  if (maxLen === 0) return null;

  return (
    <div className="section-comparison">
      <div className="freq-header">
        <span className="freq-title">Section-by-Section Comparison</span>
        <div className="freq-legend">
          <span className="legend-dot ref-dot" /> Reference
          <span className="legend-dot wip-dot" /> WIP
        </div>
      </div>
      <div className="section-cards">
        {Array.from({ length: maxLen }).map((_, i) => {
          const ref = refSecs[i];
          const wip = wipSecs[i];
          return (
            <div key={i} className="section-card">
              <div className="section-card-idx">§{i + 1}</div>
              <div className="section-card-meta">
                {ref && <span className="ref-text section-meta-label">{ref.label}<span className="section-meta-time"> {ref.start}–{ref.end}s</span></span>}
                {wip && <span className="wip-text section-meta-label">{wip.label}<span className="section-meta-time"> {wip.start}–{wip.end}s</span></span>}
              </div>
              <div className="section-loudness-row">
                <span className="section-stat-lbl">Loudness</span>
                <span className="ref-text">{ref ? `${ref.avg_loudness_db} dB` : "—"}</span>
                <span className="wip-text">{wip ? `${wip.avg_loudness_db} dB` : "—"}</span>
              </div>
              {(ref?.frequency_balance || wip?.frequency_balance) && (
                <div className="section-freq-grid">
                  {FREQ_KEYS.map((key, ki) => {
                    const rv = ref?.frequency_balance?.[key] ?? 0;
                    const wv = wip?.frequency_balance?.[key] ?? 0;
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
              )}
            </div>
          );
        })}
      </div>
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
                  ? <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>{m.content}</ReactMarkdown></div>
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

  const isAnalyzing = stage === "uploading" || stage === "extracting" || stage === "generating";

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

  const handleAnalyze = async () => {
    if (!canAnalyze) return;

    setStage("uploading");
    setSuggestions("");
    setRefAnalysis(null);
    setWipAnalysis(null);
    setPriorityScores([]);
    setJobId(null);
    setError("");

    const formData = new FormData();
    validRefs.forEach((f) => formData.append("references", f));
    formData.append("wip", wipFile);

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
            <div className="analysis-progress">
              <ProgressSteps stage={stage} />
              <div className="elapsed">{elapsed}s elapsed — large files can take 30–60 seconds</div>
            </div>
          )}
        </section>

        {/* Error */}
        {stage === "error" && (
          <div className="error-box"><span className="error-icon">⚠️</span> {error}</div>
        )}

        {/* Priority scores */}
        <PriorityScoresPanel scores={priorityScores} />

        {/* Analysis cards + frequency comparison */}
        {(refAnalysis || wipAnalysis) && stage !== "uploading" && (
          <section className="results-section">
            <h2 className="section-title">Audio Analysis</h2>
            <div className="analysis-grid">
              <AnalysisPanel analysis={refAnalysis} label={validRefs.length > 1 ? `Averaged Target (${validRefs.length} refs)` : "Reference Track"} color="#6c63ff" />
              <AnalysisPanel analysis={wipAnalysis} label="Your WIP" color="#ff6584" />
            </div>

            {refAnalysis && wipAnalysis && (
              <>
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

                <SectionComparisonPanel refAnalysis={refAnalysis} wipAnalysis={wipAnalysis} />
              </>
            )}
          </section>
        )}

        {/* AI Suggestions */}
        {(suggestions || stage === "generating") && (
          <section className="suggestions-section">
            <h2 className="section-title">
              AI Production Feedback
              {stage === "generating" && suggestions && (
                <span className="streaming-indicator">
                  <span className="pulse" /> generating...
                </span>
              )}
            </h2>
            <div className="suggestions-card">
              {suggestions ? (
                <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>{suggestions}</ReactMarkdown></div>
              ) : (
                <div className="suggestions-placeholder">
                  <span className="spinner large" />
                  <span>Waiting for AI feedback...</span>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Chat */}
        {jobId && stage === "done" && <ChatPanel jobId={jobId} />}
      </main>

      <footer className="app-footer">
        Powered by <strong>librosa</strong> + <strong>Claude Opus 4.6</strong>
      </footer>
    </div>
  );
}
