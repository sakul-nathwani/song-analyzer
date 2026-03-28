import glob
import json
import os
import time
import uuid
import tempfile
import numpy as np
import librosa
import anthropic
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(y, sr):
    # Isolate harmonic content — percussion contaminates chroma and throws off key detection
    y_harmonic = librosa.effects.harmonic(y, margin=4)

    # Average two complementary chroma representations for robustness:
    # chroma_cqt is high-resolution; chroma_cens is energy-normalised and noise-resistant
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=24)
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    chroma_mean = ((chroma_cqt + chroma_cens) / 2).mean(axis=1)

    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Z-score normalise both the profiles and the observed chroma so the
    # dot product equals the Pearson correlation — the correct K-S measure
    def _norm(v):
        std = v.std()
        return (v - v.mean()) / std if std > 1e-9 else v - v.mean()

    major_profile = _norm(major_profile)
    minor_profile = _norm(minor_profile)
    chroma_norm = _norm(chroma_mean)

    major_scores = [float(np.dot(np.roll(chroma_norm, -i), major_profile)) for i in range(12)]
    minor_scores = [float(np.dot(np.roll(chroma_norm, -i), minor_profile)) for i in range(12)]

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))

    if major_scores[best_major] >= minor_scores[best_minor]:
        return f"{NOTES[best_major]} Major"
    else:
        return f"{NOTES[best_minor]} Minor"


def detect_sections(y, sr):
    hop_length = 512

    # Combine chroma (harmonic content) + MFCCs (timbre/texture) for more
    # reliable boundary detection than chroma alone
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    features = np.vstack([
        librosa.util.normalize(chroma, axis=1),
        librosa.util.normalize(mfcc, axis=1),
    ])

    bounds = librosa.segment.agglomerative(features, 6)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)

    # Compute per-frame RMS so we can measure each section's energy level
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    sections = []
    section_energies = []

    for i in range(len(bound_times) - 1):
        start = float(bound_times[i])
        end = float(bound_times[i + 1])
        mask = (frame_times >= start) & (frame_times < end)
        avg_energy = float(np.mean(rms[mask])) if mask.any() else 0.0
        section_energies.append(avg_energy)
        sections.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "avg_energy": round(avg_energy, 6),
            "label": f"Section {i + 1}",
            "is_low_energy": False,
        })

    # Label sections relative to the track's mean energy
    if section_energies:
        mean_e = float(np.mean(section_energies))
        for i, (sec, e) in enumerate(zip(sections, section_energies)):
            ratio = e / (mean_e + 1e-9)
            if i == 0 and ratio < 0.55:
                sec["label"] = "Pre-Intro / Buildup"
                sec["is_low_energy"] = True
            elif ratio >= 1.25:
                sec["label"] = f"High-Energy Section {i + 1}"
            elif ratio < 0.65:
                sec["label"] = f"Low-Energy Section {i + 1}"
                sec["is_low_energy"] = True

    return sections


def analyze_audio(file_path: str) -> dict:
    # Resample to 22050 Hz mono — librosa's native rate and sufficient for
    # every feature we compute. Halves the array vs 44100 Hz WAV files,
    # cutting STFT/chroma/MFCC/beat-track time by ~50-60%.
    # Cap at 6 minutes so unusually long files can't time out on Railway.
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=360)
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo & BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.squeeze(tempo))

    # Key
    key = detect_key(y, sr)

    # Loudness (RMS in dBFS)
    rms = librosa.feature.rms(y=y)[0]
    avg_loudness_db = float(20 * np.log10(np.mean(rms) + 1e-9))
    peak_loudness_db = float(20 * np.log10(np.max(rms) + 1e-9))
    dynamic_range_db = float(peak_loudness_db - (20 * np.log10(np.min(rms[rms > 0]) + 1e-9)))

    # Frequency spectrum
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

    # Energy levels over time (normalized)
    hop_length = 512
    frame_length = 2048
    rms_over_time = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms_over_time)), sr=sr, hop_length=hop_length)
    # Downsample to ~20 points for readability
    step = max(1, len(rms_over_time) // 20)
    energy_profile = [
        {"time": round(float(t), 2), "energy": round(float(e), 6)}
        for t, e in zip(times[::step], rms_over_time[::step])
    ]

    # Sub-band energy distribution
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    n_fft_bins = stft.shape[0]
    sub_bass = float(np.mean(stft[freqs[:n_fft_bins] < 60, :]))
    bass = float(np.mean(stft[(freqs[:n_fft_bins] >= 60) & (freqs[:n_fft_bins] < 250), :]))
    low_mids = float(np.mean(stft[(freqs[:n_fft_bins] >= 250) & (freqs[:n_fft_bins] < 500), :]))
    mids = float(np.mean(stft[(freqs[:n_fft_bins] >= 500) & (freqs[:n_fft_bins] < 2000), :]))
    high_mids = float(np.mean(stft[(freqs[:n_fft_bins] >= 2000) & (freqs[:n_fft_bins] < 6000), :]))
    highs = float(np.mean(stft[freqs[:n_fft_bins] >= 6000, :]))
    total_energy = sub_bass + bass + low_mids + mids + high_mids + highs + 1e-9

    # Sections
    sections = detect_sections(y, sr)

    # Zero crossing rate (brightness/noisiness indicator)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    # Harmonic vs percussive balance
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio = float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-9))

    return {
        "duration_seconds": round(duration, 2),
        "tempo_bpm": round(bpm, 1),
        "key": key,
        "loudness": {
            "average_db": round(avg_loudness_db, 2),
            "peak_db": round(peak_loudness_db, 2),
            "dynamic_range_db": round(dynamic_range_db, 2),
        },
        "frequency_spectrum": {
            "spectral_centroid_hz": round(float(np.mean(spectral_centroid)), 2),
            "spectral_bandwidth_hz": round(float(np.mean(spectral_bandwidth)), 2),
            "spectral_rolloff_hz": round(float(np.mean(spectral_rolloff)), 2),
        },
        "frequency_balance": {
            "sub_bass_pct": round(sub_bass / total_energy * 100, 1),
            "bass_pct": round(bass / total_energy * 100, 1),
            "low_mids_pct": round(low_mids / total_energy * 100, 1),
            "mids_pct": round(mids / total_energy * 100, 1),
            "high_mids_pct": round(high_mids / total_energy * 100, 1),
            "highs_pct": round(highs / total_energy * 100, 1),
        },
        "energy_profile": energy_profile,
        "sections": sections,
        "zero_crossing_rate": round(zcr, 5),
        "harmonic_ratio": round(harmonic_ratio, 3),
    }


def build_comparison_prompt(ref_analysis: dict, wip_analysis: dict) -> str:
    return f"""You are an expert music producer and audio engineer. Below are the detailed audio analyses of two songs:

---
## Reference Song Analysis (the target/goal)
- Duration: {ref_analysis['duration_seconds']}s
- Tempo: {ref_analysis['tempo_bpm']} BPM
- Key: {ref_analysis['key']}
- Average Loudness: {ref_analysis['loudness']['average_db']} dBFS
- Peak Loudness: {ref_analysis['loudness']['peak_db']} dBFS
- Dynamic Range: {ref_analysis['loudness']['dynamic_range_db']} dB
- Spectral Centroid: {ref_analysis['frequency_spectrum']['spectral_centroid_hz']} Hz (brightness)
- Spectral Bandwidth: {ref_analysis['frequency_spectrum']['spectral_bandwidth_hz']} Hz
- Spectral Rolloff: {ref_analysis['frequency_spectrum']['spectral_rolloff_hz']} Hz
- Frequency Balance: Sub-bass {ref_analysis['frequency_balance']['sub_bass_pct']}% | Bass {ref_analysis['frequency_balance']['bass_pct']}% | Low-mids {ref_analysis['frequency_balance']['low_mids_pct']}% | Mids {ref_analysis['frequency_balance']['mids_pct']}% | High-mids {ref_analysis['frequency_balance']['high_mids_pct']}% | Highs {ref_analysis['frequency_balance']['highs_pct']}%
- Harmonic-to-Total Ratio: {ref_analysis['harmonic_ratio']} (higher = more melodic/tonal)
- Zero Crossing Rate: {ref_analysis['zero_crossing_rate']} (higher = noisier/more percussive)
- Number of Detected Sections: {len(ref_analysis['sections'])}
- Sections: {', '.join([f"{s['label']} ({s['start']}s–{s['end']}s, energy={s['avg_energy']:.4f})" for s in ref_analysis['sections']])}

---
## Work-in-Progress (WIP) Song Analysis
- Duration: {wip_analysis['duration_seconds']}s
- Tempo: {wip_analysis['tempo_bpm']} BPM
- Key: {wip_analysis['key']}
- Average Loudness: {wip_analysis['loudness']['average_db']} dBFS
- Peak Loudness: {wip_analysis['loudness']['peak_db']} dBFS
- Dynamic Range: {wip_analysis['loudness']['dynamic_range_db']} dB
- Spectral Centroid: {wip_analysis['frequency_spectrum']['spectral_centroid_hz']} Hz (brightness)
- Spectral Bandwidth: {wip_analysis['frequency_spectrum']['spectral_bandwidth_hz']} Hz
- Spectral Rolloff: {wip_analysis['frequency_spectrum']['spectral_rolloff_hz']} Hz
- Frequency Balance: Sub-bass {wip_analysis['frequency_balance']['sub_bass_pct']}% | Bass {wip_analysis['frequency_balance']['bass_pct']}% | Low-mids {wip_analysis['frequency_balance']['low_mids_pct']}% | Mids {wip_analysis['frequency_balance']['mids_pct']}% | High-mids {wip_analysis['frequency_balance']['high_mids_pct']}% | Highs {wip_analysis['frequency_balance']['highs_pct']}%
- Harmonic-to-Total Ratio: {wip_analysis['harmonic_ratio']} (higher = more melodic/tonal)
- Zero Crossing Rate: {wip_analysis['zero_crossing_rate']} (higher = noisier/more percussive)
- Number of Detected Sections: {len(wip_analysis['sections'])}
- Sections: {', '.join([f"{s['label']} ({s['start']}s–{s['end']}s, energy={s['avg_energy']:.4f})" for s in wip_analysis['sections']])}

---
**Important note on section interpretation:** The section detector uses energy levels to flag likely pre-intros and buildups. Any section labelled "Pre-Intro / Buildup" or "Low-Energy Section" at the very start of a track is likely a quiet fade-in, a drop/riser buildup, or a soft intro rather than the song's main intro. Do not count it as the structural intro when comparing song form. Similarly, sections labelled "High-Energy Section" are likely a chorus, drop, or climax. Use these labels as context when assessing structure and energy flow — do not assume section numbering alone maps to verse/chorus/bridge.

---
Based on these analyses, provide specific, actionable music production feedback on what is missing or underdeveloped in the WIP song compared to the reference. Structure your response with these sections:

### 🎚️ Mix & Loudness
Address differences in overall loudness, dynamic range, and how to achieve a more competitive mix level.

### 🎛️ Frequency Balance & EQ
Identify which frequency bands are over- or under-represented in the WIP compared to the reference. Give specific EQ recommendations (e.g., "boost 2–5kHz by 2–3dB to match the reference's high-mid presence").

### 🥁 Rhythm & Groove
Compare tempo, rhythmic energy, and percussive character. Suggest how to improve the groove or tightness.

### 🎵 Harmonic Content & Arrangement
Compare tonal/melodic richness and song structure. Suggest how to fill harmonic gaps or improve the arrangement.

### ✨ Brightness & Presence
Compare spectral centroid and high-frequency content. Suggest saturation, exciter, or EQ moves.

### 📐 Structure & Energy Flow
Compare how the songs build and transition. Note differences in number of sections and energy pacing.

### 🔧 Top 3 Priority Actions
List the three most impactful changes the producer should make first, ranked by importance.

Be direct, technical, and specific. Use actual numbers from the analyses when making comparisons (e.g., "the reference sits at -8dBFS average loudness while your WIP is at -14dBFS — you need to gain-stage and limit more aggressively").
"""


# ── Job store (disk-backed) ────────────────────────────────────────────────
# Each job is a JSON file at <tmpdir>/songanalyzer_<job_id>.json so results
# survive a server restart (e.g. Railway OOM-kills the process mid-analysis
# but the completed result is still readable on the next poll).
#
# Schema: {status, stage, created_at, result, error}
#   status: "running" | "done" | "error"
#   stage:  "extracting" | "generating" | "done"

_JOB_DIR = tempfile.gettempdir()
_JOB_PREFIX = "songanalyzer_"
_JOB_TTL = 3600  # 1 hour


def _job_path(job_id: str) -> str:
    return os.path.join(_JOB_DIR, f"{_JOB_PREFIX}{job_id}.json")


def _write_job(job_id: str, data: dict) -> None:
    """Atomically write job state so a crash mid-write can't corrupt the file."""
    path = _job_path(job_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp_path, path)  # atomic on POSIX


def _read_job(job_id: str) -> dict | None:
    try:
        with open(_job_path(job_id)) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _delete_job(job_id: str) -> None:
    try:
        os.unlink(_job_path(job_id))
    except OSError:
        pass


def _cleanup_old_jobs() -> None:
    """Remove job files older than _JOB_TTL seconds."""
    cutoff = time.time() - _JOB_TTL
    for path in glob.glob(os.path.join(_JOB_DIR, f"{_JOB_PREFIX}*.json")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.unlink(path)
        except OSError:
            pass


def _run_analysis(job_id: str, ref_path: str, wip_path: str) -> None:
    """Blocking worker — FastAPI runs sync BackgroundTasks in a thread pool."""
    try:
        job = _read_job(job_id) or {}
        job["stage"] = "extracting"
        _write_job(job_id, job)

        ref_analysis = analyze_audio(ref_path)
        wip_analysis = analyze_audio(wip_path)

        job["stage"] = "generating"
        _write_job(job_id, job)

        prompt = build_comparison_prompt(ref_analysis, wip_analysis)
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        suggestions = next(b.text for b in response.content if b.type == "text")

        job.update({
            "status": "done",
            "stage": "done",
            "result": {
                "reference": ref_analysis,
                "wip": wip_analysis,
                "suggestions": suggestions,
            },
        })
        _write_job(job_id, job)

    except Exception as exc:
        job = _read_job(job_id) or {}
        job.update({"status": "error", "error": str(exc)})
        _write_job(job_id, job)

    finally:
        for path in [ref_path, wip_path]:
            try:
                os.unlink(path)
            except OSError:
                pass


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    reference: UploadFile = File(...),
    wip: UploadFile = File(...),
):
    allowed_extensions = {".mp3", ".wav", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}
    for upload in [reference, wip]:
        ext = os.path.splitext(upload.filename or "")[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {upload.filename}. Supported: MP3, WAV, FLAC, OGG, AIFF, M4A",
            )

    # Stream uploads to temp files in 1 MB chunks before returning so the
    # background task has the data even after the request body is gone.
    async def _stream_to_tmp(upload: UploadFile, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            path = tmp.name
            while chunk := await upload.read(1024 * 1024):
                tmp.write(chunk)
        return path

    ref_path = await _stream_to_tmp(reference, os.path.splitext(reference.filename or ".wav")[1])
    wip_path = await _stream_to_tmp(wip, os.path.splitext(wip.filename or ".wav")[1])

    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    _write_job(job_id, {
        "status": "running",
        "stage": "extracting",
        "created_at": time.time(),
        "result": None,
        "error": None,
    })

    # Add as a sync function — FastAPI runs it in a thread pool so the
    # event loop is never blocked by librosa or the Claude API call.
    background_tasks.add_task(_run_analysis, job_id, ref_path, wip_path)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = _read_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    response = {
        "status": job["status"],   # "running" | "done" | "error"
        "stage":  job["stage"],    # "extracting" | "generating" | "done"
        "result": job["result"],
        "error":  job["error"],
    }

    # Delete the job file once the client has received the terminal result,
    # freeing disk space immediately. TTL cleanup catches anything missed.
    if job["status"] in ("done", "error"):
        _delete_job(job_id)

    return response


# ── Static file serving (production) ──────────────────────────────────────
# In dev, Vite's dev-server handles the frontend. In production (Railway),
# FastAPI serves the pre-built React app from frontend/dist.
_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)

if os.path.isdir(_DIST):
    # Serve /assets/* (Vite outputs JS/CSS bundles here)
    _assets = os.path.join(_DIST, "assets")
    if os.path.isdir(_assets):
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve exact files from dist root (favicon, icons, etc.) and fall
        back to index.html for everything else so React Router works."""
        if full_path:
            candidate = os.path.join(_DIST, full_path)
            if os.path.isfile(candidate):
                return FileResponse(candidate)
        return FileResponse(os.path.join(_DIST, "index.html"))
