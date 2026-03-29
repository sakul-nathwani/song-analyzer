import glob
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
import tempfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait as cf_wait
from datetime import datetime, timezone, timedelta
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("song-analyzer")

import numpy as np
import librosa
import anthropic
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Startup diagnostics ────────────────────────────────────────────────────
def _log_env_diagnostics() -> None:
    """Log which integration-related env vars are present (values redacted)."""
    checks = {
        "ANTHROPIC_API_KEY":    os.environ.get("ANTHROPIC_API_KEY", ""),
        "REPLICATE_API_TOKEN":  os.environ.get("REPLICATE_API_TOKEN", ""),
        "REPLICATE_API_KEY":    os.environ.get("REPLICATE_API_KEY", ""),
        "REPLICATE_TOKEN":      os.environ.get("REPLICATE_TOKEN", ""),
        "REPLICATE_KEY":        os.environ.get("REPLICATE_KEY", ""),
    }
    log.info("=== Environment variable diagnostics ===")
    for name, val in checks.items():
        if val:
            # Show first 4 chars so you can confirm which token it is without leaking it
            log.info("  %-25s SET  (starts with: %s...)", name, val[:4])
        else:
            log.info("  %-25s NOT SET", name)

    # Also list every env var that contains "replicate" (case-insensitive)
    replicate_vars = [k for k in os.environ if "replicate" in k.lower()]
    if replicate_vars:
        log.info("  All env vars containing 'replicate': %s", replicate_vars)
    else:
        log.info("  No env vars containing 'replicate' found at all")
    log.info("========================================")

_log_env_diagnostics()


def _get_replicate_token() -> str:
    """
    Read the Replicate API token, trying all known variable names Railway
    or other platforms might use.
    """
    for name in ("REPLICATE_API_TOKEN", "REPLICATE_API_KEY", "REPLICATE_TOKEN", "REPLICATE_KEY"):
        val = os.environ.get(name, "")
        if val:
            if name != "REPLICATE_API_TOKEN":
                log.info("[stems] Found Replicate token under '%s' instead of 'REPLICATE_API_TOKEN'", name)
            return val
    return ""


# ── Rate limiter (per-IP, in-memory) ───────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    detail_str = ""
    try:
        detail_str = str(exc.detail).lower()
    except Exception:
        pass
    now = datetime.now(timezone.utc)
    if "hour" in detail_str:
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        reset_str = next_hour.strftime("%I:%M %p UTC").lstrip("0") or "12:00 AM UTC"
        msg = (
            f"You've used all 10 of your free analyses for this hour. "
            f"You can run another analysis at {reset_str}. "
            "Upgrade to Pro for unlimited analyses."
        )
    else:
        msg = (
            "You've used all 30 of your free analyses for today. "
            "Your limit resets at midnight. "
            "Upgrade to Pro for unlimited analyses."
        )
    return JSONResponse(status_code=429, content={"detail": msg})

# ── Concurrency & timeout limits ────────────────────────────────────────────
_MAX_CONCURRENT_JOBS = 20
_MAX_ANALYSIS_SECONDS = 600  # 10 minutes
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB
_ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".aif"}
_job_semaphore = threading.Semaphore(_MAX_CONCURRENT_JOBS)

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
    y_harmonic = librosa.effects.harmonic(y, margin=4)
    chroma_cqt  = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=24)
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    chroma_mean = ((chroma_cqt + chroma_cens) / 2).mean(axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    def _norm(v):
        std = v.std()
        return (v - v.mean()) / std if std > 1e-9 else v - v.mean()

    major_profile = _norm(major_profile)
    minor_profile = _norm(minor_profile)
    chroma_norm   = _norm(chroma_mean)

    major_scores = [float(np.dot(np.roll(chroma_norm, -i), major_profile)) for i in range(12)]
    minor_scores = [float(np.dot(np.roll(chroma_norm, -i), minor_profile)) for i in range(12)]

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))

    if major_scores[best_major] >= minor_scores[best_minor]:
        return f"{NOTES[best_major]} Major"
    return f"{NOTES[best_minor]} Minor"


def _freq_balance_from_stft(stft_slice: np.ndarray, freqs: np.ndarray) -> dict:
    """Frequency band distribution from a pre-computed STFT magnitude slice."""
    n = stft_slice.shape[0]
    f = freqs[:n]
    sub_bass  = float(np.mean(stft_slice[f < 60]))
    bass      = float(np.mean(stft_slice[(f >= 60)  & (f < 250)]))
    low_mids  = float(np.mean(stft_slice[(f >= 250) & (f < 500)]))
    mids      = float(np.mean(stft_slice[(f >= 500) & (f < 2000)]))
    high_mids = float(np.mean(stft_slice[(f >= 2000) & (f < 6000)]))
    highs     = float(np.mean(stft_slice[f >= 6000]))
    total = sub_bass + bass + low_mids + mids + high_mids + highs + 1e-9
    return {
        "sub_bass_pct":  round(sub_bass  / total * 100, 1),
        "bass_pct":      round(bass      / total * 100, 1),
        "low_mids_pct":  round(low_mids  / total * 100, 1),
        "mids_pct":      round(mids      / total * 100, 1),
        "high_mids_pct": round(high_mids / total * 100, 1),
        "highs_pct":     round(highs     / total * 100, 1),
    }


def _heuristic_label_sections(sections: list, energies: list) -> list:
    """
    Drop-focused labeler. Identifies sustained high-energy regions as Drops (Drop 1, Drop 2, …).
    Everything else gets a generic label. Only the Drops are named precisely.

    A "Drop" is a run of consecutive segments that all meet the energy threshold and
    start after the first 20 s of the song (to exclude the intro region).
    """
    n = len(sections)
    if n == 0:
        return []

    labels  = [""] * n
    mean_e  = float(np.mean(energies))

    # Energy threshold — a segment must exceed this to be a drop candidate.
    # 1.25× the song mean filters out verses/buildups while catching typical drops.
    DROP_THRESH = mean_e * 1.25
    MIN_START   = 20.0   # ignore anything in the first 20 s (intro region)

    # 1. Collect candidates: high-energy segments that start after the intro
    candidates = [
        i for i in range(n)
        if energies[i] >= DROP_THRESH and sections[i]["start"] >= MIN_START
    ]

    # 2. Fallback: if nothing clears the threshold, pick the single loudest segment after 20 s
    if not candidates:
        after_intro = [i for i in range(n) if sections[i]["start"] >= MIN_START]
        if after_intro:
            candidates = [max(after_intro, key=lambda i: energies[i])]

    # 3. Group consecutive candidates into runs (each run becomes one Drop)
    runs: list[list[int]] = []
    if candidates:
        run = [candidates[0]]
        for c in candidates[1:]:
            if c == run[-1] + 1:
                run.append(c)
            else:
                runs.append(run)
                run = [c]
        runs.append(run)

    # 4. Sort runs by position and assign Drop labels
    runs.sort(key=lambda r: sections[r[0]]["start"])
    for drop_num, run in enumerate(runs, start=1):
        for i in run:
            labels[i] = f"Drop {drop_num}"

    # 5. First section → Intro (overrides any spurious drop assignment)
    labels[0] = "Intro"

    # 6. Last section → Outro if it's low-energy
    if n > 1 and not labels[-1] and energies[-1] < mean_e * 0.75:
        labels[-1] = "Outro"

    # 7. Everything unlabeled → generic "Section N"
    section_count = 0
    for i in range(n):
        if not labels[i]:
            section_count += 1
            labels[i] = f"Section {section_count}"

    return labels


def detect_sections(y, sr, stft=None, freqs=None, hop_length=512):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    features = np.vstack([
        librosa.util.normalize(chroma, axis=1),
        librosa.util.normalize(mfcc, axis=1),
    ])

    duration_s = len(y) / sr

    # Target ~1 segment per 15 s so short sections (16-bar buildups at 128 BPM
    # are ~30 s) aren't swallowed into adjacent segments before the AI sees them.
    # Floor: 8 segments for anything up to ~4 min so a typical
    # Intro/Verse/Buildup/Drop/Breakdown/Verse/Buildup/Drop structure is always
    # resolvable. Ceiling: 16 for longer tracks.
    n_segs = max(8, min(16, round(duration_s / 15)))
    log.info("[sections] duration=%.1fs → requesting %d segments from agglomerative", duration_s, n_segs)

    bounds      = librosa.segment.agglomerative(features, n_segs + 1)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)
    log.info("[sections] agglomerative returned %d boundaries → %d segments",
             len(bound_times), len(bound_times) - 1)

    rms         = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Pre-compute features used per-segment for AI labeling
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    onset_threshold = float(np.mean(onset_env)) * 1.5

    sections         = []
    section_energies = []

    for i in range(len(bound_times) - 1):
        start = float(bound_times[i])
        end   = float(bound_times[i + 1])
        mask  = (frame_times >= start) & (frame_times < end)
        avg_energy = float(np.mean(rms[mask])) if mask.any() else 0.0
        section_energies.append(avg_energy)

        seg_dur    = max(end - start, 1e-3)
        onset_rate = float(np.sum(onset_env[mask] > onset_threshold)) / seg_dur if mask.any() else 0.0
        avg_centroid = float(np.mean(centroid[mask])) if mask.any() else 0.0

        sec = {
            "start":        round(start, 2),
            "end":          round(end, 2),
            "avg_energy":   round(avg_energy, 6),
            "label":        f"Section {i + 1}",
            "is_low_energy": False,
            # Internal keys stripped before returning — used only for AI prompt
            "_centroid_hz": round(avg_centroid),
            "_onset_rate":  round(onset_rate, 2),
        }

        if stft is not None and freqs is not None:
            sf = max(0, librosa.time_to_frames(start, sr=sr, hop_length=hop_length))
            ef = min(stft.shape[1], librosa.time_to_frames(end, sr=sr, hop_length=hop_length))
            ef = max(ef, sf + 1)
            sec["frequency_balance"] = _freq_balance_from_stft(stft[:, sf:ef], freqs)

        y_slice = y[int(start * sr):int(end * sr)]
        if len(y_slice) > 0:
            rms_s = librosa.feature.rms(y=y_slice)[0]
            sec["avg_loudness_db"] = round(float(20 * np.log10(np.mean(rms_s) + 1e-9)), 2)
        else:
            sec["avg_loudness_db"] = -60.0

        sections.append(sec)

    # Log segment durations so merging issues are visible in the server log
    seg_summary = ", ".join(
        f"{s['start']:.0f}–{s['end']:.0f}s ({s['end']-s['start']:.0f}s)"
        for s in sections
    )
    log.info("[sections] %d segments before labeling: %s", len(sections), seg_summary)

    labels = _heuristic_label_sections(sections, section_energies)

    mean_e = float(np.mean(section_energies)) if section_energies else 1.0
    for sec, energy, label in zip(sections, section_energies, labels):
        sec["label"]         = label
        sec["is_low_energy"] = energy < mean_e * 0.75
        sec.pop("_centroid_hz", None)
        sec.pop("_onset_rate",  None)

    return sections


def detect_sidechain(y, sr, tempo_bpm, beat_frames, stft, freqs, hop_length=512):
    """
    Detect sidechain compression by looking for rhythmic bass ducking
    (energy dips just after each beat that recover before the next beat).

    Uses the 60-250 Hz range — sub-bass + bass.  The kick transient itself
    is short; sidechain compression shows as energy that dips AFTER the
    initial attack and then rises back toward the end of the beat period.
    """
    _empty = {"detected": False, "depth_db": 0.0, "release_ms": None,
              "rate": None, "consistency": 0.0}

    bass_mask = (freqs >= 60) & (freqs <= 250)
    if not bass_mask.any():
        return _empty

    bass_energy = np.mean(np.abs(stft[bass_mask, :]), axis=0)
    n_frames = len(bass_energy)

    valid = beat_frames[(beat_frames >= 0) & (beat_frames < n_frames - 1)]
    if len(valid) < 8:
        return _empty

    duck_depths    = []
    release_ms_list = []
    recovery_count = 0

    for i in range(len(valid) - 1):
        bf      = int(valid[i])
        bf_next = int(valid[i + 1])
        seg     = bass_energy[bf:bf_next]
        if len(seg) < 6:
            continue

        # Skip the first ~10 % of the beat (kick attack transient)
        skip = max(1, len(seg) // 10)
        post_attack = seg[skip:]

        if len(post_attack) < 4:
            continue

        duck_val     = float(np.min(post_attack))
        duck_rel_idx = int(np.argmin(post_attack))

        # Recovery value = mean of the latter 40 % of the beat
        recovery_start = max(duck_rel_idx + 1, int(len(post_attack) * 0.6))
        recovery_val   = float(np.mean(post_attack[recovery_start:])) if recovery_start < len(post_attack) else duck_val

        if duck_val < 1e-9 or recovery_val < 1e-9:
            continue

        depth_db = 20 * np.log10((recovery_val + 1e-9) / (duck_val + 1e-9))
        duck_depths.append(depth_db)

        if recovery_val > duck_val * 1.5:   # ≥ ~3.5 dB recovery
            recovery_count += 1

        # Release time: frames from duck minimum to 63 % of recovery
        target = duck_val + (recovery_val - duck_val) * 0.63
        abs_duck_idx = bf + skip + duck_rel_idx
        for j in range(abs_duck_idx, min(bf_next, n_frames)):
            if bass_energy[j] >= target:
                release_ms_list.append((j - abs_duck_idx) * hop_length / sr * 1000)
                break

    if not duck_depths:
        return _empty

    avg_depth    = float(np.mean(duck_depths))
    recovery_pct = recovery_count / len(duck_depths)
    detected     = avg_depth > 3.0 and recovery_pct > 0.5

    return {
        "detected":    detected,
        "depth_db":    round(avg_depth, 1),
        "release_ms":  round(float(np.mean(release_ms_list)), 0) if release_ms_list else None,
        "rate":        "1/4 note (every beat)" if detected else None,
        "consistency": round(recovery_pct, 2),
    }


def _estimate_bpm(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """
    Return (bpm, beat_frames) using a multi-method consensus:
    1. librosa.feature.tempo on the full track (onset-envelope-based global estimate)
    2. beat_track with 4/4 and 3/4 time signatures on the full track
    3. beat_track across overlapping 60-second windows (handles tempo drift)
    All candidates are collected, the median is taken as the consensus, and the
    result is rounded to the nearest 0.5 BPM.
    """
    hop_length = 512
    onset_env  = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    candidates: list[float] = []

    # ── 1. Global tempo via autocorrelation (librosa.feature.tempo) ──────────
    global_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    candidates.append(float(np.squeeze(global_tempo)))

    # ── 2. beat_track with 4/4 (tightness=100) and 3/4 (tightness=80) ───────
    tempo_44, beat_frames_44 = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, tightness=100
    )
    candidates.append(float(np.squeeze(tempo_44)))

    tempo_34, _ = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        tightness=80, start_bpm=float(np.squeeze(tempo_44)) * (3 / 4),
    )
    candidates.append(float(np.squeeze(tempo_34)) * (4 / 3))  # normalise to 4/4 equivalent

    # ── 3. Windowed estimates (60-second windows, 30-second step) ────────────
    window_samples = int(60 * sr)
    step_samples   = int(30 * sr)
    for start in range(0, max(1, len(y) - window_samples), step_samples):
        segment = y[start : start + window_samples]
        if len(segment) < sr * 10:   # skip segments shorter than 10 s
            continue
        env_seg = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=hop_length)
        t, _ = librosa.beat.beat_track(
            onset_envelope=env_seg, sr=sr, hop_length=hop_length, tightness=100
        )
        val = float(np.squeeze(t))
        if val > 0:
            candidates.append(val)

    # ── Consensus: median, then round to nearest 0.5 BPM ────────────────────
    consensus = float(np.median([c for c in candidates if c > 0]))
    bpm = round(consensus * 2) / 2   # nearest 0.5

    # Return beat_frames from the full-track 4/4 pass (used downstream)
    return bpm, beat_frames_44


def analyze_audio(file_path: str) -> dict:
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=360)
    duration = librosa.get_duration(y=y, sr=sr)

    bpm, beat_frames = _estimate_bpm(y, sr)

    key = detect_key(y, sr)

    rms              = librosa.feature.rms(y=y)[0]
    avg_loudness_db  = float(20 * np.log10(np.mean(rms) + 1e-9))
    peak_loudness_db = float(20 * np.log10(np.max(rms) + 1e-9))
    dynamic_range_db = float(peak_loudness_db - (20 * np.log10(np.min(rms[rms > 0]) + 1e-9)))

    spectral_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

    hop_length    = 512
    frame_length  = 2048
    rms_over_time = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times         = librosa.frames_to_time(np.arange(len(rms_over_time)), sr=sr, hop_length=hop_length)

    # Smooth energy profile: sliding window of ~3 s, step of ~1 s.
    # This eliminates per-frame noise and produces a clean, readable curve.
    fps            = sr / hop_length                        # frames per second (~43.1)
    win_frames     = max(1, int(3.0 * fps))                 # 3-second averaging window
    step_frames    = max(1, int(1.0 * fps))                 # 1 point per second
    energy_profile = []
    for i in range(0, len(rms_over_time), step_frames):
        w_end       = min(i + win_frames, len(rms_over_time))
        center_idx  = min(i + win_frames // 2, len(times) - 1)
        avg_energy  = float(np.mean(rms_over_time[i:w_end]))
        energy_profile.append({
            "time":   round(float(times[center_idx]), 2),
            "energy": round(avg_energy, 6),
        })

    # Compute STFT once — reused for overall + per-section frequency analysis
    stft  = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    overall_freq_balance = _freq_balance_from_stft(stft, freqs)

    sections  = detect_sections(y, sr, stft=stft, freqs=freqs, hop_length=hop_length)
    sidechain = detect_sidechain(y, sr, bpm, beat_frames, stft, freqs, hop_length)

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    y_harmonic, _ = librosa.effects.hpss(y)
    harmonic_ratio = float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-9))

    return {
        "duration_seconds": round(duration, 2),
        "tempo_bpm":        bpm,  # already rounded to nearest 0.5 by _estimate_bpm
        "key":              key,
        "loudness": {
            "average_db":       round(avg_loudness_db, 2),
            "peak_db":          round(peak_loudness_db, 2),
            "dynamic_range_db": round(dynamic_range_db, 2),
        },
        "frequency_spectrum": {
            "spectral_centroid_hz":  round(float(np.mean(spectral_centroid)), 2),
            "spectral_bandwidth_hz": round(float(np.mean(spectral_bandwidth)), 2),
            "spectral_rolloff_hz":   round(float(np.mean(spectral_rolloff)), 2),
        },
        "frequency_balance":  overall_freq_balance,
        "energy_profile":     energy_profile,
        "sections":           sections,
        "sidechain":          sidechain,
        "zero_crossing_rate": round(zcr, 5),
        "harmonic_ratio":     round(harmonic_ratio, 3),
    }


# ── Stem separation (Replicate) ────────────────────────────────────────────

_STEM_NAMES = ["drums", "bass", "other"]


def _analyze_stem(path: str, stem_name: str) -> dict:
    """Lightweight librosa analysis of a single separated stem file."""
    y, sr = librosa.load(path, sr=22050, mono=True, duration=360)
    hop_length = 512
    stft  = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    freq_balance = _freq_balance_from_stft(stft, freqs)
    rms     = librosa.feature.rms(y=y)[0]
    avg_db  = float(20 * np.log10(np.mean(rms) + 1e-9))
    peak_db = float(20 * np.log10(np.max(rms) + 1e-9))
    dyn_db  = float(peak_db - (20 * np.log10(np.min(rms[rms > 0]) + 1e-9)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
    return {
        "stem": stem_name,
        "loudness": {
            "average_db":       round(avg_db, 2),
            "peak_db":          round(peak_db, 2),
            "dynamic_range_db": round(dyn_db, 2),
        },
        "frequency_balance":    freq_balance,
        "spectral_centroid_hz": round(centroid, 2),
    }


def _get_or_create_stems(file_path: str, file_hash: str) -> tuple[dict, str | None]:
    """
    Return (stem_path_map, error_message).
    stem_path_map is {stem_name: local_tmp_path} for drums / bass / other, or {}
    on failure. error_message is None on success, a human-readable string on failure.

    Passes the audio file directly to the Replicate SDK (which handles its own
    upload internally), then downloads the returned stems to local temp files for
    librosa analysis. No external storage is used.
    """
    replicate_token = _get_replicate_token()
    if not replicate_token:
        msg = (
            "No Replicate API token found — checked REPLICATE_API_TOKEN, REPLICATE_API_KEY, "
            "REPLICATE_TOKEN, REPLICATE_KEY. Set one of these in your environment."
        )
        log.warning("[stems] %s", msg)
        return {}, msg

    log.info("[stems] Starting stem pipeline for hash=%s", file_hash[:12])

    # ── 1. Run Demucs on Replicate, passing the file object directly ───────
    try:
        import replicate as _replicate  # noqa: PLC0415
        log.info("[stems] Submitting to Replicate (ryan5453/demucs) — this may take 1-3 minutes")
        t0 = time.time()
        # Pin the version hash so the SDK calls the prediction endpoint directly
        # rather than resolving the latest version (which can itself 404).
        MODEL = "ryan5453/demucs:5a7041cc9b82e5a558fea6b3d7b12dea89625e89da33f0447bd727c2d0ab9e77"
        with open(file_path, "rb") as audio_fh:
            if os.environ.get("REPLICATE_API_TOKEN", "") != replicate_token:
                replicate_client = _replicate.Client(api_token=replicate_token)
                output = replicate_client.run(MODEL, input={"audio": audio_fh})
            else:
                output = _replicate.run(MODEL, input={"audio": audio_fh})
        elapsed = time.time() - t0
        log.info("[stems] Replicate completed in %.1fs, output type=%s", elapsed, type(output).__name__)
        log.info("[stems] Raw output: %r", str(output)[:300])
    except Exception as exc:
        msg = f"Replicate API call failed: {exc}"
        log.error("[stems] %s", msg, exc_info=True)
        return {}, msg

    # ── 2. Parse output into {stem_name: url} ─────────────────────────────
    # ryan5453/demucs returns {"stems": [{name, audio}, ...]}
    stem_url_map: dict[str, str] = {}
    stems_list = None
    if isinstance(output, dict):
        stems_list = output.get("stems")
    # Some SDK versions materialise the output as an object with a .stems attribute
    if stems_list is None and hasattr(output, "stems"):
        stems_list = output.stems
    if stems_list:
        for item in stems_list:
            if isinstance(item, dict):
                name, url = item.get("name"), item.get("audio") or item.get("url")
            else:
                name = getattr(item, "name", None)
                url  = getattr(item, "audio", None) or getattr(item, "url", None)
            if name and url and name in _STEM_NAMES:
                stem_url_map[name] = str(url)
    # Fallback: flat dict {stem_name: url}
    if not stem_url_map and isinstance(output, dict):
        stem_url_map = {k: str(v) for k, v in output.items() if k in _STEM_NAMES}
    log.info("[stems] Stems parsed from output: %s", list(stem_url_map.keys()))

    if not stem_url_map:
        msg = f"Replicate returned no recognisable stem URLs. Raw output: {str(output)[:300]}"
        log.error("[stems] %s", msg)
        return {}, msg

    # ── 3. Download each stem to a local temp file for librosa ────────────
    stem_paths: dict[str, str] = {}
    for stem in _STEM_NAMES:
        url = stem_url_map.get(stem)
        if not url:
            log.warning("[stems] No URL for stem '%s' in Replicate output", stem)
            continue
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                stem_local = tf.name
            log.info("[stems] Downloading stem '%s'...", stem)
            urllib.request.urlretrieve(url, stem_local)
            log.info("[stems] Downloaded '%s': %d bytes", stem, os.path.getsize(stem_local))
            stem_paths[stem] = stem_local
        except Exception as exc:
            log.error("[stems] Failed to download stem '%s': %s", stem, exc, exc_info=True)

    if not stem_paths:
        return {}, "All stem downloads from Replicate failed"

    log.info("[stems] Ready to analyse %d stems: %s", len(stem_paths), list(stem_paths.keys()))
    return stem_paths, None


def _run_stem_analysis_pipeline(file_path: str, file_hash: str) -> tuple[dict, str | None]:
    """
    Fetch/create stems then run librosa on each in parallel.
    Returns (analysis_dict, error_message). On failure analysis_dict is {}.
    """
    stem_paths, stem_error = _get_or_create_stems(file_path, file_hash)
    if not stem_paths:
        return {}, stem_error

    log.info("[stems] Running librosa on %d stems in parallel", len(stem_paths))
    results: dict[str, dict] = {}
    try:
        with ThreadPoolExecutor(max_workers=len(stem_paths)) as pool:
            futures = {stem: pool.submit(_analyze_stem, path, stem)
                       for stem, path in stem_paths.items()}
            for stem, fut in futures.items():
                try:
                    results[stem] = fut.result(timeout=120)
                    log.info("[stems] librosa analysis done for stem '%s'", stem)
                except Exception as exc:
                    log.error("[stems] librosa analysis failed for stem '%s': %s", stem, exc)
    finally:
        for p in stem_paths.values():
            try: os.unlink(p)
            except: pass

    if not results:
        return {}, "Stem analysis (librosa) failed for all stems"
    return results, None


def _average_stem_analyses(analyses_list: list[dict]) -> dict:
    """Average per-stem analysis dicts from multiple reference files."""
    all_stems: set[str] = set()
    for a in analyses_list:
        all_stems.update(a.keys())
    result: dict[str, dict] = {}
    for stem in all_stems:
        dicts = [a[stem] for a in analyses_list if stem in a]
        if not dicts:
            continue
        n = len(dicts)
        result[stem] = {
            "stem": stem,
            "loudness": {
                k: round(sum(d["loudness"][k] for d in dicts) / n, 2)
                for k in dicts[0]["loudness"]
            },
            "frequency_balance": {
                k: round(sum(d["frequency_balance"][k] for d in dicts) / n, 1)
                for k in dicts[0]["frequency_balance"]
            },
            "spectral_centroid_hz": round(
                sum(d["spectral_centroid_hz"] for d in dicts) / n, 2
            ),
        }
    return result


# ── Analysis cache ─────────────────────────────────────────────────────────
# Keyed by SHA-256 of file contents. Re-uploading the same audio file skips
# all librosa work and returns the stored result instantly.

_analysis_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 20  # max entries; oldest evicted first (FIFO) when full


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _analyze_cached(path: str) -> dict:
    """Run analyze_audio, returning a cached result if the file was seen before."""
    digest = _file_hash(path)
    with _cache_lock:
        if digest in _analysis_cache:
            return _analysis_cache[digest]

    result = analyze_audio(path)

    with _cache_lock:
        if len(_analysis_cache) >= _CACHE_MAX:
            del _analysis_cache[next(iter(_analysis_cache))]  # evict oldest
        _analysis_cache[digest] = result

    return result


def _average_analysis_dicts(analyses: list) -> dict:
    """Average multiple analyses into a single target reference profile."""
    n = len(analyses)
    if n == 1:
        return analyses[0]

    def avg_scalar(key):
        return round(sum(a[key] for a in analyses) / n, 2)

    def avg_subdict(key):
        return {k: round(sum(a[key][k] for a in analyses) / n, 2) for k in analyses[0][key]}

    # Average sidechain across all references
    sidechains = [a.get("sidechain") for a in analyses if a.get("sidechain")]
    if sidechains:
        sc_n = len(sidechains)
        detected_count = sum(1 for s in sidechains if s.get("detected"))
        avg_sc = {
            "detected":    detected_count >= sc_n / 2,
            "depth_db":    round(sum(s["depth_db"] for s in sidechains) / sc_n, 1),
            "release_ms":  (
                round(sum(s["release_ms"] for s in sidechains if s.get("release_ms") is not None) /
                      max(1, sum(1 for s in sidechains if s.get("release_ms") is not None)))
                if any(s.get("release_ms") is not None for s in sidechains) else None
            ),
            "rate":        Counter(s["rate"] for s in sidechains if s.get("rate")).most_common(1)[0][0]
                           if any(s.get("rate") for s in sidechains) else None,
            "consistency": round(sum(s["consistency"] for s in sidechains) / sc_n, 2),
        }
    else:
        avg_sc = {"detected": False, "depth_db": 0.0, "release_ms": None, "rate": None, "consistency": 0.0}

    return {
        "duration_seconds":   avg_scalar("duration_seconds"),
        "tempo_bpm":          round(sum(a["tempo_bpm"] for a in analyses) / n * 2) / 2,
        "key":                Counter(a["key"] for a in analyses).most_common(1)[0][0],
        "loudness":           avg_subdict("loudness"),
        "frequency_spectrum": avg_subdict("frequency_spectrum"),
        "frequency_balance":  avg_subdict("frequency_balance"),
        "energy_profile":     analyses[0]["energy_profile"],
        "sections":           analyses[0]["sections"],
        "sidechain":          avg_sc,
        "zero_crossing_rate": round(sum(a["zero_crossing_rate"] for a in analyses) / n, 5),
        "harmonic_ratio":     round(sum(a["harmonic_ratio"] for a in analyses) / n, 3),
    }


def _parse_priority_scores(text: str) -> list:
    m = re.search(r"<priority_scores>\s*(.*?)\s*</priority_scores>", text, re.DOTALL)
    if not m:
        return []
    try:
        scores = json.loads(m.group(1))
        if isinstance(scores, list):
            return scores[:3]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _build_stem_prompt_section(ref_stems: dict | None, wip_stems: dict | None) -> str:
    if not ref_stems and not wip_stems:
        return ""

    def _fmt(s: dict) -> str:
        fb = s.get("frequency_balance", {})
        return (
            f"avg {s['loudness']['average_db']} dBFS | peak {s['loudness']['peak_db']} dBFS | "
            f"DR {s['loudness']['dynamic_range_db']} dB | centroid {s['spectral_centroid_hz']} Hz | "
            f"Sub {fb.get('sub_bass_pct','?')}% Bass {fb.get('bass_pct','?')}% "
            f"LM {fb.get('low_mids_pct','?')}% M {fb.get('mids_pct','?')}% "
            f"HM {fb.get('high_mids_pct','?')}% Hi {fb.get('highs_pct','?')}%"
        )

    lines = ["\n---\n## Stem Analysis (Deep Mode — Demucs separation)\n"]
    for stem in _STEM_NAMES:
        ref_s = (ref_stems or {}).get(stem)
        wip_s = (wip_stems or {}).get(stem)
        if ref_s or wip_s:
            lines.append(f"\n**{stem.title()}**")
            if ref_s:
                lines.append(f"- Reference: {_fmt(ref_s)}")
            if wip_s:
                lines.append(f"- WIP:       {_fmt(wip_s)}")
    lines.append(
        "\nUse the stem-level data to give specific per-element feedback "
        "(e.g. bass EQ, drum transient punch, arrangement level balance between stems)."
    )
    return "\n".join(lines)


def build_comparison_prompt(
    ref_analysis: dict,
    wip_analysis: dict,
    n_refs: int = 1,
    ref_stems: dict | None = None,
    wip_stems: dict | None = None,
) -> str:
    ref_label = f"Averaged Target ({n_refs} references)" if n_refs > 1 else "Reference Track"

    def fmt_sections(analysis):
        lines = []
        for s in analysis["sections"]:
            fb = s.get("frequency_balance", {})
            fb_str = (
                f" | Sub {fb.get('sub_bass_pct','?')}% Bass {fb.get('bass_pct','?')}%"
                f" LM {fb.get('low_mids_pct','?')}% M {fb.get('mids_pct','?')}%"
                f" HM {fb.get('high_mids_pct','?')}% Hi {fb.get('highs_pct','?')}%"
            ) if fb else ""
            loud = f" | {s.get('avg_loudness_db','?')} dBFS" if "avg_loudness_db" in s else ""
            lines.append(f"  - {s['label']} ({s['start']}s–{s['end']}s, energy={s['avg_energy']:.4f}){fb_str}{loud}")
        return "\n".join(lines)

    def fmt_sidechain(sc):
        if not sc:
            return "No data"
        status = "DETECTED" if sc.get("detected") else "not detected"
        parts = [status, f"depth {sc['depth_db']} dB"]
        if sc.get("release_ms") is not None:
            parts.append(f"release {sc['release_ms']} ms")
        if sc.get("rate"):
            parts.append(f"rate: {sc['rate']}")
        parts.append(f"consistency {round(sc.get('consistency', 0) * 100)}%")
        return " | ".join(parts)

    return f"""You are an expert music producer and audio engineer.

IMPORTANT: Begin your response with a priority scores block, then the full markdown analysis.

<priority_scores>
[
  {{"score": <1-10 integer, 10=most critical>, "label": "<3-5 word issue name>", "summary": "<one sentence: what is wrong and how to fix it>"}},
  ...exactly 3 items, sorted by score descending — the 3 most impactful issues only
]
</priority_scores>

---
## {ref_label}
- Duration: {ref_analysis['duration_seconds']}s | Tempo: {ref_analysis['tempo_bpm']} BPM | Key: {ref_analysis['key']}
- Loudness: avg {ref_analysis['loudness']['average_db']} dBFS | peak {ref_analysis['loudness']['peak_db']} dBFS | DR {ref_analysis['loudness']['dynamic_range_db']} dB
- Spectral centroid: {ref_analysis['frequency_spectrum']['spectral_centroid_hz']} Hz | BW: {ref_analysis['frequency_spectrum']['spectral_bandwidth_hz']} Hz | Rolloff: {ref_analysis['frequency_spectrum']['spectral_rolloff_hz']} Hz
- Freq balance: Sub {ref_analysis['frequency_balance']['sub_bass_pct']}% | Bass {ref_analysis['frequency_balance']['bass_pct']}% | LowMids {ref_analysis['frequency_balance']['low_mids_pct']}% | Mids {ref_analysis['frequency_balance']['mids_pct']}% | HiMids {ref_analysis['frequency_balance']['high_mids_pct']}% | Highs {ref_analysis['frequency_balance']['highs_pct']}%
- Sidechain: {fmt_sidechain(ref_analysis.get('sidechain'))}
- Harmonic ratio: {ref_analysis['harmonic_ratio']} | ZCR: {ref_analysis['zero_crossing_rate']}
- Sections ({len(ref_analysis['sections'])} detected):
{fmt_sections(ref_analysis)}

---
## WIP Track
- Duration: {wip_analysis['duration_seconds']}s | Tempo: {wip_analysis['tempo_bpm']} BPM | Key: {wip_analysis['key']}
- Loudness: avg {wip_analysis['loudness']['average_db']} dBFS | peak {wip_analysis['loudness']['peak_db']} dBFS | DR {wip_analysis['loudness']['dynamic_range_db']} dB
- Spectral centroid: {wip_analysis['frequency_spectrum']['spectral_centroid_hz']} Hz | BW: {wip_analysis['frequency_spectrum']['spectral_bandwidth_hz']} Hz | Rolloff: {wip_analysis['frequency_spectrum']['spectral_rolloff_hz']} Hz
- Freq balance: Sub {wip_analysis['frequency_balance']['sub_bass_pct']}% | Bass {wip_analysis['frequency_balance']['bass_pct']}% | LowMids {wip_analysis['frequency_balance']['low_mids_pct']}% | Mids {wip_analysis['frequency_balance']['mids_pct']}% | HiMids {wip_analysis['frequency_balance']['high_mids_pct']}% | Highs {wip_analysis['frequency_balance']['highs_pct']}%
- Sidechain: {fmt_sidechain(wip_analysis.get('sidechain'))}
- Harmonic ratio: {wip_analysis['harmonic_ratio']} | ZCR: {wip_analysis['zero_crossing_rate']}
- Sections ({len(wip_analysis['sections'])} detected):
{fmt_sections(wip_analysis)}

---
Section note: labels are EDM-structure heuristics (Intro/Verse/Buildup/Drop/Breakdown/Outro) derived from relative energy and position. Treat them as starting context, not ground truth — reference actual energy and frequency values when giving feedback.

---
After the <priority_scores> block, provide your full analysis structured as:

### 🎚️ Mix & Loudness
### 🎛️ Frequency Balance & EQ
### 🥁 Rhythm & Groove
### 🔊 Sidechain & Compression
### 🎵 Harmonic Content & Arrangement
### ✨ Brightness & Presence
### 📐 Structure & Energy Flow
### 🔧 Top 3 Priority Actions

Be direct, technical, and specific. Use actual numbers from the analyses.
""" + _build_stem_prompt_section(ref_stems, wip_stems)


# ── Job store (disk-backed) ────────────────────────────────────────────────

_JOB_DIR    = tempfile.gettempdir()
_JOB_PREFIX = "songanalyzer_"
_JOB_TTL    = 3600  # 1 hour


def _job_path(job_id: str) -> str:
    return os.path.join(_JOB_DIR, f"{_JOB_PREFIX}{job_id}.json")


def _write_job(job_id: str, data: dict) -> None:
    path = _job_path(job_id)
    tmp  = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


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
    cutoff = time.time() - _JOB_TTL
    for path in glob.glob(os.path.join(_JOB_DIR, f"{_JOB_PREFIX}*.json")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.unlink(path)
        except OSError:
            pass


def _run_analysis(job_id: str, ref_paths: list, wip_path: str, n_refs: int, deep_analysis: bool = False) -> None:
    """Blocking worker — FastAPI runs sync BackgroundTasks in a thread pool."""
    deadline = time.time() + _MAX_ANALYSIS_SECONDS
    try:
        job = _read_job(job_id) or {}
        job["stage"] = "extracting"
        _write_job(job_id, job)

        all_paths = ref_paths + [wip_path]
        stem_error_msg: str | None = None
        log.info("[job %s] Starting analysis: %d file(s), deep_analysis=%s", job_id[:8], len(all_paths), deep_analysis)

        # Compute hashes up-front when stem caching is needed
        all_hashes = [_file_hash(p) for p in all_paths] if deep_analysis else []
        if deep_analysis:
            log.info("[job %s] File hashes: %s", job_id[:8], [h[:12] for h in all_hashes])

        max_workers = len(all_paths) * (2 if deep_analysis else 1)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            analysis_futures = [pool.submit(_analyze_cached, p) for p in all_paths]
            stem_futures = (
                [pool.submit(_run_stem_analysis_pipeline, p, h)
                 for p, h in zip(all_paths, all_hashes)]
                if deep_analysis else []
            )

            # Update stage to "separating" once audio analysis is done but stems are still running
            if stem_futures:
                audio_remaining = max(0.1, deadline - time.time())
                audio_done, audio_not_done = cf_wait(analysis_futures, timeout=audio_remaining)
                if audio_not_done:
                    for f in audio_not_done:
                        f.cancel()
                    raise TimeoutError(
                        "Audio analysis timed out after 10 minutes. "
                        "Please try a shorter or smaller audio file."
                    )
                job["stage"] = "separating"
                _write_job(job_id, job)
                log.info("[job %s] Audio analysis done, waiting for stem separation", job_id[:8])

                stem_remaining = max(0.1, deadline - time.time())
                stem_done, stem_not_done = cf_wait(stem_futures, timeout=stem_remaining)
                if stem_not_done:
                    log.warning("[job %s] Stem futures timed out, continuing without stems", job_id[:8])
                    for f in stem_not_done:
                        f.cancel()
            else:
                all_futures = analysis_futures
                remaining = max(0.1, deadline - time.time())
                done, not_done = cf_wait(all_futures, timeout=remaining)
                if not_done:
                    for f in not_done:
                        f.cancel()
                    raise TimeoutError(
                        "Audio analysis timed out after 10 minutes. "
                        "Please try a shorter or smaller audio file."
                    )

            results = [f.result() for f in analysis_futures]

            # Collect stem results — each future returns (analysis_dict, error_msg)
            stem_errors: list[str] = []
            raw_stem_results: list[dict] = []
            for fut in stem_futures:
                if fut.done() and not fut.cancelled():
                    try:
                        s_analysis, s_err = fut.result()
                        raw_stem_results.append(s_analysis)
                        if s_err:
                            stem_errors.append(s_err)
                    except Exception as exc:
                        raw_stem_results.append({})
                        stem_errors.append(str(exc))
                else:
                    raw_stem_results.append({})
                    stem_errors.append("Stem separation timed out")

            stem_results = raw_stem_results if raw_stem_results else [{} for _ in all_paths]

        ref_analyses   = results[: len(ref_paths)]
        wip_analysis   = results[-1]
        ref_analysis   = _average_analysis_dicts(ref_analyses)
        log.info("[job %s] Audio analysis complete", job_id[:8])

        ref_stem_list  = stem_results[: len(ref_paths)]
        wip_stems      = stem_results[-1] if stem_results else {}
        ref_stems      = _average_stem_analyses(ref_stem_list) if any(ref_stem_list) else {}

        # Summarise stem outcome
        stem_error_msg: str | None = None
        if deep_analysis:
            if ref_stems or wip_stems:
                log.info("[job %s] Stem separation succeeded — ref stems: %s, wip stems: %s",
                         job_id[:8], list(ref_stems.keys()), list(wip_stems.keys()))
            else:
                stem_error_msg = stem_errors[0] if stem_errors else "Stem separation produced no results"
                log.warning("[job %s] Stem separation failed: %s", job_id[:8], stem_error_msg)

        # ── Generate AI feedback ─────────────────────────────────────────────
        if time.time() > deadline:
            raise TimeoutError(
                "Audio analysis timed out before AI feedback could be generated. "
                "Please try again with a shorter audio file."
            )

        job = _read_job(job_id) or {}
        if job.get("status") == "cancelled":
            log.info("[job %s] Cancelled before AI call — exiting", job_id[:8])
            return

        job["stage"] = "generating"
        _write_job(job_id, job)

        stem_analyses_result = None
        if deep_analysis and (ref_stems or wip_stems):
            stem_analyses_result = {"reference": ref_stems, "wip": wip_stems}

        prompt = build_comparison_prompt(
            ref_analysis, wip_analysis, n_refs=n_refs,
            ref_stems=ref_stems or None,
            wip_stems=wip_stems or None,
        )
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = next(b.text for b in response.content if b.type == "text")

        priority_scores = _parse_priority_scores(raw_text)
        suggestions = re.sub(
            r"<priority_scores>.*?</priority_scores>\s*", "", raw_text, flags=re.DOTALL
        ).strip()

        log.info("[job %s] Analysis complete — writing result", job_id[:8])
        job.update({
            "status": "done",
            "stage":  "done",
            "result": {
                "reference":       ref_analysis,
                "wip":             wip_analysis,
                "suggestions":     suggestions,
                "priority_scores": priority_scores,
                "n_refs":          n_refs,
                "stem_analyses":   stem_analyses_result,
                "stem_error":      stem_error_msg,
            },
        })
        _write_job(job_id, job)

    except Exception as exc:
        job = _read_job(job_id) or {}
        job.update({"status": "error", "error": str(exc)})
        _write_job(job_id, job)

    finally:
        _job_semaphore.release()
        for path in ref_paths + [wip_path]:
            try:
                os.unlink(path)
            except OSError:
                pass


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/analyze")
@limiter.limit("10/hour")
@limiter.limit("30/day")
async def analyze(
    request: Request,
    background_tasks: BackgroundTasks,
    references: List[UploadFile] = File(...),
    wip: UploadFile = File(...),
    deep_analysis: bool = Form(False),
):
    # Concurrency cap
    if not _job_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail=(
                f"MixRef is currently at capacity with {_MAX_CONCURRENT_JOBS} active analyses. "
                "Please try again in a few minutes."
            ),
        )

    try:
        if not references or len(references) > 3:
            raise HTTPException(status_code=400, detail="Provide 1–3 reference tracks")

        for upload in list(references) + [wip]:
            ext = os.path.splitext(upload.filename or "")[1].lower()
            if ext not in _ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type: {upload.filename}. "
                        "Accepted formats: MP3, WAV, FLAC, AIFF"
                    ),
                )

        async def _stream_to_tmp(upload: UploadFile, suffix: str) -> str:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                path = tmp.name
                total = 0
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > _MAX_FILE_BYTES:
                        tmp.close()
                        os.unlink(path)
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"'{upload.filename}' exceeds the 100 MB size limit. "
                                "Please upload a shorter audio file."
                            ),
                        )
                    tmp.write(chunk)
            return path

        ref_paths = [
            await _stream_to_tmp(ref, os.path.splitext(ref.filename or ".wav")[1])
            for ref in references
        ]
        wip_path = await _stream_to_tmp(wip, os.path.splitext(wip.filename or ".wav")[1])

    except HTTPException:
        # Release semaphore if we won't be handing off to the background task
        _job_semaphore.release()
        raise

    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    _write_job(job_id, {
        "status":     "running",
        "stage":      "extracting",
        "created_at": time.time(),
        "result":     None,
        "error":      None,
    })

    # Background task owns the semaphore from here; it will release in its finally block
    background_tasks.add_task(_run_analysis, job_id, ref_paths, wip_path, len(references), deep_analysis)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = _read_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    # No auto-delete — job file must persist for the /chat endpoint.
    # TTL cleanup (_cleanup_old_jobs) handles removal after 1 hour.
    result = job["result"] or {}
    return {
        "status":     job["status"],
        "stage":      job["stage"],
        "result":     job["result"],
        "error":      job["error"],
        "stem_error": result.get("stem_error"),
    }


@app.post("/cancel/{job_id}")
def cancel_job(job_id: str):
    job = _read_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job.get("status") not in ("running",):
        return {"status": "ok"}  # already done/errored — nothing to do
    job["status"] = "cancelled"
    job["stage"]  = "cancelled"
    _write_job(job_id, job)
    log.info("[job %s] Cancelled by user", job_id[:8])
    return {"status": "ok"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.post("/chat/{job_id}")
def chat(job_id: str, request: ChatRequest):
    job = _read_job(job_id)
    if not job or job["status"] != "done" or not job.get("result"):
        raise HTTPException(status_code=404, detail="Analysis not found or not yet complete")

    result = job["result"]
    stem_ctx = ""
    if result.get("stem_analyses"):
        stem_ctx = f"\nSTEM ANALYSES (Demucs):\n{json.dumps(result['stem_analyses'], indent=2)}\n"
    system = (
        "You are an expert music producer and audio engineer assistant.\n"
        "The user has analyzed two tracks. Here is the full analysis data:\n\n"
        f"REFERENCE:\n{json.dumps(result['reference'], indent=2)}\n\n"
        f"WIP:\n{json.dumps(result['wip'], indent=2)}\n"
        f"{stem_ctx}\n"
        f"AI FEEDBACK ALREADY PROVIDED:\n{result['suggestions']}\n\n"
        f"PRIORITY ISSUES:\n{json.dumps(result.get('priority_scores', []), indent=2)}\n\n"
        "Answer the user's follow-up questions concisely and technically. "
        "Reference actual numbers from the analysis. "
        "When recommending plugins, name specific ones (e.g. FabFilter Pro-Q 3, Ozone, Waves SSL)."
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
    )
    return {"reply": response.content[0].text}


# ── Static file serving (production) ──────────────────────────────────────

_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)

if os.path.isdir(_DIST):
    _assets = os.path.join(_DIST, "assets")
    if os.path.isdir(_assets):
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path:
            candidate = os.path.join(_DIST, full_path)
            if os.path.isfile(candidate):
                return FileResponse(candidate)
        return FileResponse(os.path.join(_DIST, "index.html"))
