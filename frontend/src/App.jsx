import { useState, useRef, useCallback } from "react";
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

export default function App() {
  const [refFile, setRefFile] = useState(null);
  const [wipFile, setWipFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | analyzing | done | error
  const [suggestions, setSuggestions] = useState("");
  const [refAnalysis, setRefAnalysis] = useState(null);
  const [wipAnalysis, setWipAnalysis] = useState(null);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!refFile || !wipFile) return;

    setStatus("analyzing");
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
        const err = await res.json();
        throw new Error(err.detail || "Analysis failed");
      }

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
            } else if (msg.type === "text") {
              setSuggestions((prev) => prev + msg.content);
            } else if (msg.type === "done") {
              setStatus("done");
            }
          } catch {
            // ignore parse errors on partial chunks
          }
        }
      }

      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  };

  const canAnalyze = refFile && wipFile && status !== "analyzing";

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
            className={`analyze-btn ${canAnalyze ? "active" : ""} ${status === "analyzing" ? "loading" : ""}`}
            onClick={handleAnalyze}
            disabled={!canAnalyze}
          >
            {status === "analyzing" ? (
              <>
                <span className="spinner" />
                Analyzing...
              </>
            ) : (
              <>
                <span className="btn-icon">✦</span>
                Run Analysis
              </>
            )}
          </button>
        </section>

        {/* Error */}
        {status === "error" && (
          <div className="error-box">
            <span className="error-icon">⚠️</span> {error}
          </div>
        )}

        {/* Analysis Cards */}
        {(refAnalysis || wipAnalysis) && (
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
        {(suggestions || status === "analyzing") && (
          <section className="suggestions-section">
            <h2 className="section-title">
              AI Production Feedback
              {status === "analyzing" && suggestions && (
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
                  <span>Extracting features and generating feedback...</span>
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
