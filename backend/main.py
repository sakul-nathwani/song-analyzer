import glob
import hashlib
import json
import os
import re
import threading
import time
import uuid
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait as cf_wait
from datetime import datetime, timezone, timedelta
from typing import List

import numpy as np
import librosa
import anthropic
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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


def detect_sections(y, sr, stft=None, freqs=None, hop_length=512):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    features = np.vstack([
        librosa.util.normalize(chroma, axis=1),
        librosa.util.normalize(mfcc, axis=1),
    ])

    # Scale target sections to track length: ~1 section per 45 s, clamped 4–8.
    # Each track sizes independently so ref and WIP can have different counts.
    duration_s = len(y) / sr
    n_segs     = max(4, min(8, round(duration_s / 45)))
    bounds      = librosa.segment.agglomerative(features, n_segs + 1)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)

    rms         = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    sections         = []
    section_energies = []

    for i in range(len(bound_times) - 1):
        start = float(bound_times[i])
        end   = float(bound_times[i + 1])
        mask  = (frame_times >= start) & (frame_times < end)
        avg_energy = float(np.mean(rms[mask])) if mask.any() else 0.0
        section_energies.append(avg_energy)

        sec = {
            "start":      round(start, 2),
            "end":        round(end, 2),
            "avg_energy": round(avg_energy, 6),
            "label":      f"Section {i + 1}",
            "is_low_energy": False,
        }

        # Per-section frequency balance — free since STFT is already computed
        if stft is not None and freqs is not None:
            sf = max(0, librosa.time_to_frames(start, sr=sr, hop_length=hop_length))
            ef = min(stft.shape[1], librosa.time_to_frames(end, sr=sr, hop_length=hop_length))
            ef = max(ef, sf + 1)
            sec["frequency_balance"] = _freq_balance_from_stft(stft[:, sf:ef], freqs)

        # Per-section loudness
        y_slice = y[int(start * sr):int(end * sr)]
        if len(y_slice) > 0:
            rms_s = librosa.feature.rms(y=y_slice)[0]
            sec["avg_loudness_db"] = round(float(20 * np.log10(np.mean(rms_s) + 1e-9)), 2)
        else:
            sec["avg_loudness_db"] = -60.0

        sections.append(sec)

    if section_energies:
        mean_e = float(np.mean(section_energies))
        n      = len(sections)
        labels = [""] * n

        # Pass 0 — First section is ALWAYS Intro in EDM, regardless of energy.
        # A track never opens with a Drop; a high-energy opener is still an intro.
        labels[0] = "Intro"

        # Pass 1 — Outro: last section clearly below average energy (≥2 sections needed)
        if n > 1 and section_energies[-1] < mean_e * 0.75:
            labels[-1] = "Outro"

        # Pass 2 — Drop detection with three structural guards:
        #   (a) must start ≥30 s into the song  — drops never open a track
        #   (b) must be ≥20 s long              — filters transient spikes
        #   (c) context-aware energy threshold:
        #       · 1.35× mean when preceded by a plausible buildup (unlabeled section
        #         with moderate energy ≥0.85× mean — it has "build" energy without
        #         yet being a drop itself)
        #       · 1.50× mean otherwise (more conservative without a clear buildup)
        for i in range(n):
            if labels[i]:
                continue
            if sections[i]["start"] < 30.0:
                continue
            duration = sections[i]["end"] - sections[i]["start"]
            has_buildup_before = (
                i > 0
                and not labels[i - 1]
                and section_energies[i - 1] >= mean_e * 0.85
            )
            threshold = mean_e * 1.35 if has_buildup_before else mean_e * 1.50
            if section_energies[i] >= threshold and duration >= 20.0:
                labels[i] = "Drop"

        # Fallback: if still no drops, relax only the duration guard while keeping
        # the position guard (≥30 s) and the context-aware energy thresholds.
        if not any(lbl == "Drop" for lbl in labels):
            for i in range(n):
                if labels[i]:
                    continue
                if sections[i]["start"] < 30.0:
                    continue
                has_buildup_before = (
                    i > 0
                    and not labels[i - 1]
                    and section_energies[i - 1] >= mean_e * 0.85
                )
                threshold = mean_e * 1.25 if has_buildup_before else mean_e * 1.40
                if section_energies[i] >= threshold:
                    labels[i] = "Drop"

        # Pass 3 — Buildup: unlabeled section immediately before a Drop
        for i in range(n - 1):
            if not labels[i] and labels[i + 1] == "Drop":
                labels[i] = "Buildup"

        # Pass 4 — Breakdown: unlabeled section after a Drop with clearly lower energy
        for i in range(1, n):
            if not labels[i] and labels[i - 1] == "Drop":
                drop_e = section_energies[i - 1]
                if section_energies[i] < drop_e * 0.82:
                    labels[i] = "Breakdown"

        # Pass 5 — Verse: everything remaining
        for i in range(n):
            if not labels[i]:
                labels[i] = "Verse"

        for sec, e, label in zip(sections, section_energies, labels):
            sec["label"]         = label
            sec["is_low_energy"] = e < mean_e * 0.75

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


def analyze_audio(file_path: str) -> dict:
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=360)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.squeeze(tempo))

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
        "tempo_bpm":        round(bpm, 1),
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
        "tempo_bpm":          round(sum(a["tempo_bpm"] for a in analyses) / n, 1),
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
            return scores
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def build_comparison_prompt(ref_analysis: dict, wip_analysis: dict, n_refs: int = 1) -> str:
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
  ...6-8 items sorted by score descending
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
"""


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


def _run_analysis(job_id: str, ref_paths: list, wip_path: str, n_refs: int) -> None:
    """Blocking worker — FastAPI runs sync BackgroundTasks in a thread pool."""
    deadline = time.time() + _MAX_ANALYSIS_SECONDS
    try:
        job = _read_job(job_id) or {}
        job["stage"] = "extracting"
        _write_job(job_id, job)

        all_paths = ref_paths + [wip_path]
        with ThreadPoolExecutor(max_workers=len(all_paths)) as pool:
            futures_list = [pool.submit(_analyze_cached, p) for p in all_paths]
            remaining = max(0.1, deadline - time.time())
            done, not_done = cf_wait(futures_list, timeout=remaining)
            if not_done:
                for f in not_done:
                    f.cancel()
                raise TimeoutError(
                    "Audio analysis timed out after 10 minutes. "
                    "Please try a shorter or smaller audio file."
                )
            results = [f.result() for f in futures_list]

        ref_analyses = results[: len(ref_paths)]
        wip_analysis = results[-1]
        ref_analysis = _average_analysis_dicts(ref_analyses)

        if time.time() > deadline:
            raise TimeoutError(
                "Analysis timed out before AI feedback could be generated. "
                "Please try again with a shorter audio file."
            )

        job["stage"] = "generating"
        _write_job(job_id, job)

        prompt = build_comparison_prompt(ref_analysis, wip_analysis, n_refs=n_refs)
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

        job.update({
            "status": "done",
            "stage":  "done",
            "result": {
                "reference":       ref_analysis,
                "wip":             wip_analysis,
                "suggestions":     suggestions,
                "priority_scores": priority_scores,
                "n_refs":          n_refs,
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
    background_tasks.add_task(_run_analysis, job_id, ref_paths, wip_path, len(references))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = _read_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    # No auto-delete — job file must persist for the /chat endpoint.
    # TTL cleanup (_cleanup_old_jobs) handles removal after 1 hour.
    return {
        "status": job["status"],
        "stage":  job["stage"],
        "result": job["result"],
        "error":  job["error"],
    }


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
    system = (
        "You are an expert music producer and audio engineer assistant.\n"
        "The user has analyzed two tracks. Here is the full analysis data:\n\n"
        f"REFERENCE:\n{json.dumps(result['reference'], indent=2)}\n\n"
        f"WIP:\n{json.dumps(result['wip'], indent=2)}\n\n"
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
