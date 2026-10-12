"""Self-check for api.py: `python test_api.py` or `pytest`."""
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api import app

SAMPLES = Path(__file__).parent / "samples"
client = TestClient(app)


def wait(job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/transcriptions/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_upload_then_poll():
    with open(SAMPLES / "short.mp3", "rb") as f:
        r = client.post("/v1/transcriptions", files={"file": ("short.mp3", f)})
    assert r.status_code == 202, r.text
    job = wait(r.json()["id"])
    assert job["status"] == "done", job
    assert "quick brown fox" in job["result"]["text"].lower()
    assert job["segments"] == job["result"]["segments"]  # streamed segments equal the final ones


def test_bad_upload_fails_without_retry():
    r = client.post("/v1/transcriptions", files={"file": ("notes.txt", b"this is not audio")})
    assert r.status_code == 202
    job = wait(r.json()["id"])
    assert job["status"] == "failed" and "cannot decode" in job["error"], job


def test_unknown_job_is_404():
    assert client.get("/v1/transcriptions/nope").status_code == 404


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
