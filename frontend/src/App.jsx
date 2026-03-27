import { useState, useRef, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

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

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };
  const handleDragLeave = () => setDragging(false);

  const handleClick = () => inputRef.current?.click();

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
          <span className="file-size">
            ({(file.size / 1024 / 1024).toFixed(1)} MB)
          </span>
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

function FreqBar({ label, refPct, wipPct }) {
  return (
    <div className="freq-bar-row">
      <div className="freq-bar-label">{label}</div>
      <div className="freq-bars">
        <div className="freq-bar-track">
          <div
            className="freq-bar ref-bar"
            style={{ width: `${Math.min(refPct * 4, 100)}%` }}
            title={`Ref: ${refPct}%`}
          />
        </div>
        <div className="freq-bar-track">
          <div
            className="freq-bar wip-bar"
            style={{ width: `${Math.min(wipPct * 4, 100)}%` }}
            title={`WIP: ${wipPct}%`}
          />
        </div>
      </div>
      <div className="freq-bar-values">
        <span className="ref-text">{refPct}%</span>
        <span className="wip-text">{wipPct}%</span>
      </div>
    </div>
  );
}

function AnalysisPanel({ analysis, label, color }) {
  if (!analysis) return null;
  const fb = analysis.frequency_balance;

  return (
    <div className="analysis-panel" style={{ "--accent": color }}>
      <div className="panel-header">
        <span className="panel-dot" />
        {label}
      </div>
      <div className="stats-grid">
        <StatCard label="Duration" value={analysis.duration_seconds} unit="s" />
        <StatCard label="Tempo" value={analysis.tempo_bpm} unit="BPM" />
        <StatCard label="Key" value={analysis.key} />
        <StatCard
          label="Avg Loudness"
          value={analysis.loudness.average_db}
          unit="dBFS"
        />
        <StatCard
          label="Peak Loudness"
          value={analysis.loudness.peak_db}
          unit="dBFS"
        />
        <StatCard
          label="Dynamic Range"
          value={analysis.loudness.dynamic_range_db}
          unit="dB"
        />
        <StatCard
          label="Spectral Centroid"
          value={Math.round(analysis.frequency_spectrum.spectral_centroid_hz)}
          unit="Hz"
          sub="brightness"
        />
        <StatCard
          label="Harmonic Ratio"
          value={analysis.harmonic_ratio}
          sub="tonal richness"
        />
        <StatCard
          label="Sections"
          value={analysis.sections.length}
          sub="detected"
        />
      </div>
    </div>
  );
}

// stage: idle → uploading → extracting → generating → done | error
const STAGE_LABELS = {
  idle:       null,
  uploading:  "Uploading files...",
  extracting: "Extracting audio features...",
  generating: "Generating AI feedback...",
  done:       null,
  error:      null,
};

const STEPS = [
  { key: "uploading",  label: "Upload" },
  { key: "extracting", label: "Analyze Audio" },
  { key: "generating", label: "AI Feedback" },
];

function ProgressSteps({ stage }) {
  const order = ["uploading", "extracting", "generating", "done"];
  const current = order.indexOf(stage);
  return (
    <div className="progress-steps">
      {STEPS.map((step, i) => {
        const stepIdx = i; // 0=uploading,1=extracting,2=generating
        const done = current > stepIdx || stage === "done";
        const active = current === stepIdx && stage !== "done";
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

export default function App() {
  const [refFile, setRefFile] = useState(null);
  const [wipFile, setWipFile] = useState(null);
  const [stage, setStage] = useState("idle");
  const [suggestions, setSuggestions] = useState("");
  const [refAnalysis, setRefAnalysis] = useState(null);
  const [wipAnalysis, setWipAnalysis] = useState(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

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

  const handleAnalyze = async () => {
    if (!refFile || !wipFile) return;

    setStage("uploading");
    setSuggestions("");
    setRefAnalysis(null);
    setWipAnalysis(null);
    setError("");

    const formData = new FormData();
    formData.append("reference", refFile);
    formData.append("wip", wipFile);

    try {
      const res = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        // Gracefully handle non-JSON error bodies (e.g. Railway 502 HTML pages)
        let msg = `Server error ${res.status}`;
        try {
          const body = await res.json();
          msg = body.detail || msg;
        } catch {
          try { msg = (await res.text()).slice(0, 200) || msg; } catch {}
        }
        throw new Error(msg);
      }

      setStage("extracting");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          try {
            const msg = JSON.parse(raw);
            if (msg.type === "analysis") {
              setRefAnalysis(msg.reference);
              setWipAnalysis(msg.wip);
              setStage("generating");
            } else if (msg.type === "text") {
              setSuggestions((prev) => prev + msg.content);
            } else if (msg.type === "done") {
              setStage("done");
            } else if (msg.type === "error") {
              throw new Error(msg.message || "Analysis failed on server");
            }
          } catch (parseErr) {
            if (parseErr.message !== parseErr.message) return; // re-throw real errors
            if (parseErr instanceof SyntaxError) continue; // partial SSE chunk
            throw parseErr;
          }
        }
      }

      if (stage !== "done") setStage("done");
    } catch (err) {
      setError(err.message);
      setStage("error");
    }
  };

  const canAnalyze = refFile && wipFile && !isAnalyzing;

  const freqLabels = [
    ["Sub-bass", "sub_bass_pct"],
    ["Bass", "bass_pct"],
    ["Low-mids", "low_mids_pct"],
    ["Mids", "mids_pct"],
    ["High-mids", "high_mids_pct"],
    ["Highs", "highs_pct"],
  ];

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🎧</span>
            <span className="logo-text">Song Analyzer</span>
          </div>
          <p className="header-subtitle">
            AI-powered music production feedback — compare your WIP to a reference track
          </p>
        </div>
      </header>

      <main className="app-main">
        {/* Upload Section */}
        <section className="upload-section">
          <div className="upload-grid">
            <AudioUploadBox
              label="Reference Track"
              sublabel="The song you're aiming for"
              file={refFile}
              onFileChange={setRefFile}
              color="#6c63ff"
              icon="🎯"
            />
            <div className="upload-vs">VS</div>
            <AudioUploadBox
              label="Your WIP"
              sublabel="Your work-in-progress"
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
              <>
                <span className="spinner" />
                {STAGE_LABELS[stage]}
              </>
            ) : (
              <>
                <span className="btn-icon">✦</span>
                Run Analysis
              </>
            )}
          </button>

          {isAnalyzing && (
            <div className="analysis-progress">
              <ProgressSteps stage={stage} />
              <div className="elapsed">
                {elapsed}s elapsed — large files can take 30–60 seconds
              </div>
            </div>
          )}
        </section>

        {/* Error */}
        {stage === "error" && (
          <div className="error-box">
            <span className="error-icon">⚠️</span> {error}
          </div>
        )}

        {/* Analysis Cards */}
        {(refAnalysis || wipAnalysis) && stage !== "uploading" && (
          <section className="results-section">
            <h2 className="section-title">Audio Analysis</h2>

            <div className="analysis-grid">
              <AnalysisPanel
                analysis={refAnalysis}
                label="Reference Track"
                color="#6c63ff"
              />
              <AnalysisPanel
                analysis={wipAnalysis}
                label="Your WIP"
                color="#ff6584"
              />
            </div>

            {/* Frequency comparison */}
            {refAnalysis && wipAnalysis && (
              <div className="freq-comparison">
                <div className="freq-header">
                  <span className="freq-title">Frequency Balance Comparison</span>
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
                <div className="markdown-body">
                  <ReactMarkdown>{suggestions}</ReactMarkdown>
                </div>
              ) : (
                <div className="suggestions-placeholder">
                  <span className="spinner large" />
                  <span>Waiting for AI feedback...</span>
                </div>
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
