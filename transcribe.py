"""Audio -> text with per-segment timestamps, as JSON for downstream use.

Usage:
    python transcribe.py samples/short.mp3                 # JSON to stdout
    python transcribe.py samples/long.mp3 -o out.json      # JSON to file
    python transcribe.py call.wav --model medium --device cuda --batch-size 8

Any container/codec FFmpeg can read is accepted (wav, mp3, m4a, ogg, flac, mp4 ...);
decoding and resampling to 16 kHz mono happen in-process via PyAV.
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

SAMPLE_RATE = 16_000


@functools.lru_cache(maxsize=2)
def get_model(model: str, device: str, compute_type: str) -> WhisperModel:
    """Loading weights costs seconds; a long-lived service pays it once."""
    return WhisperModel(model, device=device, compute_type=compute_type)


def load_audio(path: Path):
    """Decode any audio/video file to 16 kHz mono float32. Fails fast on bad input."""
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    try:
        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    except Exception as e:  # PyAV raises many types; the message is what matters
        raise ValueError(f"cannot decode {path.name}: {e}") from e
    if audio.size == 0:
        raise ValueError(f"{path.name} contains no audio")
    return audio


def transcribe(
    path: str | Path,
    model: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    batch_size: int = 1,
    on_segment=None,
) -> dict:
    """Return {"segments": [{"id", "start", "end", "text"}], "text", ...}.

    Segments are consumed lazily from the model, so `on_segment` can stream
    results (log, websocket, DB insert) before a long file has finished.
    """
    path = Path(path)
    audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE

    whisper = get_model(model, device, compute_type)
    t0 = time.perf_counter()
    opts = {"language": language, "vad_filter": True, "word_timestamps": False}
    if batch_size > 1:
        # batched mode: VAD cuts speech into chunks that run as one batch (GPU win)
        segments, info = BatchedInferencePipeline(whisper).transcribe(
            audio, batch_size=batch_size, **opts
        )
    else:
        segments, info = whisper.transcribe(audio, **opts)

    out = []
    for i, s in enumerate(segments):
        # the last window is zero-padded to 30 s and the model may place its end
        # timestamp inside that padding, so clamp to the real duration
        seg = {"id": i, "start": round(s.start, 2), "end": round(min(s.end, duration), 2),
               "text": s.text.strip()}
        out.append(seg)
        if on_segment:
            on_segment(seg)

    return {
        "file": path.name,
        "duration_sec": round(duration, 2),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "model": model,
        "processing_sec": round(time.perf_counter() - t0, 2),
        "segments": out,
        "text": " ".join(s["text"] for s in out),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path)
    p.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    p.add_argument("--model", default="small", help="tiny|base|small|medium|large-v3|distil-large-v3 or a CT2 model path")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--compute-type", default="int8", help="int8 runs everywhere; float16 is faster on GPU")
    p.add_argument("--language", help="ISO code, e.g. en, ur. Omit to auto-detect")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("-q", "--quiet", action="store_true", help="do not print segments as they arrive")
    a = p.parse_args(argv)

    def show(seg):
        print(f"[{seg['start']:8.2f} -> {seg['end']:8.2f}] {seg['text']}", file=sys.stderr)

    try:
        result = transcribe(
            a.audio, a.model, a.device, a.compute_type, a.language, a.batch_size,
            on_segment=None if a.quiet else show,
        )
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.output:
        a.output.write_text(text, encoding="utf-8")
        print(f"wrote {a.output} ({len(result['segments'])} segments, "
              f"{result['duration_sec']}s audio in {result['processing_sec']}s)", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
