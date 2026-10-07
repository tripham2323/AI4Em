"""Rebuild the project's original, sample-free alert chime (standard library only)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import wave


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "assets" / "warning.wav"
SAMPLE_RATE = 24_000
AMPLITUDE = 0.25
PULSES = ((660, 0.18), (0, 0.08), (880, 0.18))
FADE_SECONDS = 0.008


def synthesize() -> bytes:
    """Create two soft sine chimes with short silence and click-free edges."""
    frames = bytearray()
    for frequency, duration in PULSES:
        count = round(SAMPLE_RATE * duration)
        for index in range(count):
            if frequency == 0:
                sample = 0.0
            else:
                fade = min(1.0, index / (SAMPLE_RATE * FADE_SECONDS),
                           (count - index - 1) / (SAMPLE_RATE * FADE_SECONDS))
                sample = AMPLITUDE * max(0.0, fade) * math.sin(
                    2 * math.pi * frequency * index / SAMPLE_RATE
                )
            frames.extend(struct.pack("<h", round(sample * 32767)))
    return bytes(frames)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pcm = synthesize()
    with wave.open(str(OUTPUT), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    metadata = {
        "asset": "warning.wav",
        "purpose": "Neutral two-tone drowsiness alert chime; no speech or third-party samples.",
        "source": "Generated deterministically by scripts/generate_warning_asset.py using Python standard library math and wave.",
        "license": "CC0-1.0",
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "sample_format": "signed 16-bit PCM",
        "duration_seconds": round(len(pcm) / (SAMPLE_RATE * 2), 3),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    OUTPUT.with_suffix(".wav.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {OUTPUT}")
    print(f"Duration: {metadata['duration_seconds']}s; SHA-256: {metadata['sha256']}")


if __name__ == "__main__":
    main()
