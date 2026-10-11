# Transcription pipeline

Audio file in, JSON with per-segment timestamps out. Built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), the CTranslate2 port of
OpenAI Whisper. Runs fully offline on CPU. No external `ffmpeg` binary and no PyTorch.

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on Linux/macOS
pip install -r requirements.txt
python transcribe.py samples/short.mp3               # JSON on stdout, segments stream to stderr
python transcribe.py samples/long.mp3 -o long.json   # 9.7 min file
python test_transcribe.py                            # or: pytest
```

The first run downloads the model (`small`, about 480 MB) from the Hugging Face Hub.

## Output

```json
{
  "file": "short.mp3", "duration_sec": 11.9, "language": "en", "language_probability": 0.999,
  "model": "small", "processing_sec": 5.6,
  "segments": [
    {"id": 0, "start": 0.0, "end": 5.08, "text": "Hello, this is a short test recording for the transcription pipeline."},
    {"id": 1, "start": 5.08, "end": 8.52, "text": "The quick brown fox jumps over the lazy dog."}
  ],
  "text": "Hello, this is a short test recording ... with timestamps."
}
```

`transcribe()` is importable. Pass `on_segment=` to receive each segment as soon as the model
emits it, so a service can stream partial results for a long file instead of waiting for the end.

## Options

| Flag | Default | Why |
|------|---------|-----|
| `--model` | `small` | best accuracy per CPU-second for English; `medium`/`large-v3` for harder audio, `tiny`/`base` for speed |
| `--device` | `cpu` | `cuda` when a GPU with cuBLAS/cuDNN is present |
| `--compute-type` | `int8` | runs everywhere; `float16` on GPU |
| `--language` | auto | fix it (`en`, `ur`) when you know it; saves the detection pass and avoids wrong-language output on short clips |
| `--batch-size` | 1 | `>1` switches to the batched pipeline: VAD cuts speech into chunks that are decoded together (large speed-up on GPU) |

## Design decisions

### Engine: faster-whisper, not the reference Whisper or a cloud API

Whisper is the most widely used open-weights speech model: multilingual (99 languages), robust
to accents and noise, and it emits timestamp tokens natively. Open weights mean no per-minute
cost, no audio leaving the machine, and tests that run offline. faster-whisper runs the same
weights on CTranslate2, a C++ inference engine, giving the same accuracy at roughly 4x the CPU
speed with int8 quantisation and no PyTorch dependency. It also ships Silero VAD and a batched
pipeline, so the whole service is one dependency and about 120 lines. A hosted API (Deepgram,
AssemblyAI, OpenAI) would win on diarization and streaming at the cost of data leaving the
premises; that trade is worth revisiting if those become requirements.

### Model size: `small` by default

Best accuracy per CPU-second on clean English. `base` is twice as fast and was enough for the
synthetic samples; `medium` or `large-v3` for accents, noise or Urdu. It is a flag, and the
right answer is to measure word error rate on real audio and pick.

### Audio formats: decide by content, not extension

`faster_whisper.decode_audio` uses PyAV, which bundles the FFmpeg libraries. FFmpeg probes the
container by content, decodes the codec, and the resampler converts to 16 kHz mono float32,
the only input Whisper understands. One code path for WAV, MP3, M4A/AAC, OGG/Opus, FLAC, WebM
and the audio track of MP4/MKV. No subprocess, no `ffmpeg` on PATH, one pip install. A file
that is missing, will not decode, or decodes to nothing fails with a one-line error before
the model loads. Stereo is averaged to mono; for telephony with one speaker per channel,
decode channels separately and tag the segments.

### Timestamps: from the model, clamped to the file

Whisper's decoder emits timestamp tokens at 20 ms resolution, so two decimals is honest
precision. The last 30-second window is zero-padded and the model sometimes places its end
token inside the padding; the first test run produced a segment ending at 17.28 s on an
11.9 s file. `end` is therefore clamped to the decoded duration, and the self-check asserts
monotonic, non-overlapping segments inside the file. Word-level timestamps are one flag away
(`word_timestamps=True`, cross-attention DTW); WhisperX forced alignment is the upgrade for
subtitle-grade precision.

### Long audio: sliding window, VAD, streaming

Whisper only ever sees 30 seconds. faster-whisper slides the window across the file and
carries the previous window's text as a prompt so sentences that cross a boundary stay
coherent. Three choices matter for long files:

1. **VAD on by default.** Silero VAD drops silence before decoding: less work, and it removes
   Whisper's main failure on long files, repeated or invented text over silent stretches.
2. **Segments are a generator.** Results are emitted as they arrive, so memory does not grow
   with transcript length and a caller sees progress from the first seconds.
3. **Batched mode** (`--batch-size 8`) decodes VAD chunks in parallel; large speed-up on GPU.

Decoded audio costs 3.84 MB per minute (16 kHz float32), so one hour is about 230 MB in RAM.
Beyond a few hours, split the file at silence and transcribe the pieces with a time offset.

### Measured (laptop CPU, int8, model `small`)

| File | Audio | Mode | Segments | Processing | Real-time factor |
|------|-------|------|----------|-----------|------------------|
| short.mp3 | 11.9 s | sequential | 3 | 5.6 s | 0.47 |
| long.mp3 | 584.7 s | sequential | 138 (sentences) | 617 s | 1.05 |
| long.mp3 | 584.7 s | `--batch-size 8` | 20 (VAD chunks of ~28 s) | 84 s | 0.14 |

Both long runs transcribed all 40 paragraphs correctly. Batched mode trades sentence-level
segments for speed; the sample's heavily repeated text also triggers Whisper's temperature
fallback in sequential mode, which inflates that number relative to natural speech.

### Testing

Samples in `samples/` are synthetic (Windows text-to-speech), so ground truth is known and
there is no copyright or privacy concern. `test_transcribe.py` transcribes the same sentence
from four containers and asserts identical text, English detected, and valid timestamps; it
checks the streaming callback sees every segment; and it checks a missing file and a non-audio
file fail cleanly. Real evaluation needs real recordings and a WER measurement (`jiwer`).

### Dependency pin

`av<16`: PyAV 16 removed the `metadata_errors` keyword that faster-whisper 1.2.1 still passes.

### Not built on purpose

HTTP wrapper, job queue, speaker diarization, SRT/VTT export. Each is a thin layer on top of
`transcribe()`; the system design below describes them.

## System design

How this becomes a service. Nothing below is implemented; the transcription logic stays as is.

### Concurrent uploads

Transcription never runs inside an HTTP request. `POST /v1/transcriptions` creates a job row
(status `queued`) and returns a job id plus a pre-signed object-storage URL; the client uploads
directly to S3 or MinIO, so the API never buffers large files and a burst of uploads is a
burst of PUTs to storage. On upload completion the job id goes on a queue (SQS, Redis, or a
Postgres `SELECT ... FOR UPDATE SKIP LOCKED` table). Workers each load the model once and take
one job at a time, so concurrency equals worker count, chosen from CPU/GPU memory rather than
by traffic; queue depth is the autoscaling signal. Idempotency key per upload (client key or
content hash) prevents duplicate jobs; size and duration limits are enforced at the upload
URL; job creation is rate-limited per API key. On GPU workers, many short files are decoded
together with the batched pipeline.

### Storage

- **Audio**: object storage, keyed by job id, in the original format (decode at run time;
  nothing lost if a better model or word timestamps are wanted later). Encrypted at rest,
  reachable only through short-lived pre-signed URLs, deleted after an agreed retention
  period while the transcript is kept.
- **Jobs** (Postgres): id, owner, status, audio key, duration, language, model name, attempt
  count, last committed offset, created/updated, last error.
- **Segments** (Postgres): job id, index, start, end, text, unique on (job id, index) so writes
  are idempotent. Rows rather than a JSON blob so segments can be searched (trigram or
  full-text index), edited individually, joined to speakers later, and written incrementally
  while a long file is still running.
- **Export**: final JSON written next to the audio in object storage for consumers that
  only want the file. Model name is stored per job because transcripts from different model
  sizes are not equal, and it identifies what to re-run after an upgrade.

### Retry and recovery

The job row is a state machine: `queued`, `running`, `done`, `failed`. A worker takes a job
with a lease (SQS visibility timeout, or a `locked_until` column) and heartbeats; if the
worker dies the lease expires and the job is visible again. Error classes are handled
differently: input errors (will not decode, zero duration, over limit) fail immediately and
never retry; transient errors (OOM, storage timeout, lost worker) retry with exponential
backoff up to three attempts, then move to a dead-letter state with the error on the job row
and an alert, with a manual requeue endpoint. Attempt count lives on the job so the limit
survives worker restarts. Segments are written as produced along with the last committed
audio offset, so a retry resumes from that offset and the unique constraint makes overlap
harmless; with very long files split at silence into offset chunks, a failure costs one
chunk. A periodic sweeper returns jobs stuck in `running` past any sane duration to the queue.

### API

Versioned REST, asynchronous by default, FastAPI so the OpenAPI document is generated from
the code.

| Method and path | Purpose |
|-----------------|---------|
| `POST /v1/transcriptions` | create job; body: language hint, model, webhook URL; returns 202, job id, upload URL, status URL |
| `GET /v1/transcriptions/{id}` | status, progress (last committed timestamp / duration), language and full text when done |
| `GET /v1/transcriptions/{id}/segments?format=json\|srt\|vtt` | segments in the consumer's shape |
| `GET /v1/transcriptions/{id}/events` | server-sent events: each segment as the worker commits it |
| `DELETE /v1/transcriptions/{id}` | remove audio and transcript (privacy requests) |
| `POST /v1/transcriptions:sync` | convenience for clips under a minute, hard timeout |

Webhook on completion or failure, signed so the receiver can verify it. API key per customer,
rate limits per key, consistent error bodies with a machine-readable code.
