"""Self-check: `python test_transcribe.py` or `pytest`. Uses the synthetic samples/."""
from pathlib import Path

from transcribe import transcribe

SAMPLES = Path(__file__).parent / "samples"
EXPECTED = "quick brown fox"


def check(result):
    segs = result["segments"]
    assert segs, "no segments returned"
    assert EXPECTED in result["text"].lower(), result["text"]
    assert result["language"] == "en"
    last_end = 0.0
    for s in segs:
        assert 0 <= s["start"] <= s["end"] <= result["duration_sec"] + 0.5, s
        assert s["start"] >= last_end, f"segments overlap: {s}"
        last_end = s["end"]


def test_every_format_gives_same_transcript():
    texts = set()
    for f in ["short.wav", "short.mp3", "short.m4a", "short.ogg"]:
        r = transcribe(SAMPLES / f, model="base")
        check(r)
        texts.add(r["text"].lower())
    assert len(texts) == 1, f"formats disagree: {texts}"


def test_streaming_callback_sees_every_segment():
    seen = []
    r = transcribe(SAMPLES / "short.wav", model="base", on_segment=seen.append)
    assert seen == r["segments"]


def test_bad_input_fails_clearly():
    for bad in [SAMPLES / "missing.wav", Path(__file__)]:  # absent file, non-audio file
        try:
            transcribe(bad, model="base")
        except (FileNotFoundError, ValueError) as e:
            assert bad.name in str(e)
        else:
            raise AssertionError(f"{bad.name} should have been rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
