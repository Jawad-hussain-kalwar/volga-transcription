"""Smallest HTTP front for transcribe(), shaped like the design in README: accept fast,
work in the background, poll for segments.

    uvicorn api:app
    curl -F file=@samples/short.mp3 localhost:8000/v1/transcriptions     # 202 {"id": ..., "status": "queued"}
    curl localhost:8000/v1/transcriptions/<id>                           # status, segments so far, result when done
"""
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from transcribe import transcribe

app = FastAPI(title="transcription")

# shortcut: in-memory job table and a one-thread queue stand in for Postgres + a real queue,
# swap them before running more than one process
JOBS: dict[str, dict] = {}
WORKER = ThreadPoolExecutor(max_workers=1)


def run(job_id: str, path: Path):
    job = JOBS[job_id]
    job["status"] = "running"
    try:
        job["result"] = transcribe(path, on_segment=job["segments"].append)
        job["status"] = "done"
    except (FileNotFoundError, ValueError) as e:  # input error: permanent, never retried
        job["status"], job["error"] = "failed", str(e)
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


@app.post("/v1/transcriptions", status_code=202)
def create(file: UploadFile):
    # shortcut: temp dir stands in for object storage; the upload streams to disk, never into RAM
    path = Path(tempfile.mkdtemp()) / Path(file.filename or "audio").name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"id": job_id, "file": path.name, "status": "queued", "segments": []}
    WORKER.submit(run, job_id, path)
    return {"id": job_id, "status": "queued", "url": f"/v1/transcriptions/{job_id}"}


@app.get("/v1/transcriptions/{job_id}")
def get(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    return JOBS[job_id]
