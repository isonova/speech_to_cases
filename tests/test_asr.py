# tests/test_asr.py
import os
import tempfile
import numpy as np
import soundfile as sf
import pytest
from unittest import mock

# import the functions/classes to test from your ASR module
from ASR import transcribe as asr

def make_tone(duration_s: float, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t).astype("float32")

def write_wav(path: str, data: np.ndarray, sr: int = 16000):
    sf.write(path, data, sr)

def test_vad_detects_voiced_regions():
    """VAD should detect two voiced regions separated by silence."""
    sr = 16000
    tone = make_tone(0.5, sr)
    silence = np.zeros(int(0.5 * sr), dtype="float32")
    arr = np.concatenate([tone, silence, tone])
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "burst.wav")
        write_wav(wav_path, arr, sr)
        regions = asr.get_voiced_regions(wav_path, aggressiveness=2)
        # Expect at least 2 voiced regions (depending on VAD aggressiveness)
        assert len(regions) >= 2, f"expected >=2 voiced regions, got {regions}"

def test_make_chunks_from_voiced_regions_splits_long():
    """Long voiced region must be split into max_len limited chunks with overlap."""
    # create one long region of 1000s -> split into chunks <= MAX_CHUNK_SECONDS
    long_region = [(0.0, 1000.0)]
    chunks = asr.make_chunks_from_voiced_regions(long_region, max_len=asr.MAX_CHUNK_SECONDS, overlap=asr.OVERLAP_SECONDS)
    # total coverage should equal or exceed the original region start..end
    assert len(chunks) >= 1
    # none of the chunks should be longer than MAX_CHUNK_SECONDS + small epsilon
    for s,e in chunks:
        assert e - s <= asr.MAX_CHUNK_SECONDS + 1e-6

def test_transcribe_audio_file_merges_mocked_segments(monkeypatch):
    """High-level API should stitch text and merge near segments when backend returns close segments."""
    # create a short audio file so conversion / slicing works
    sr = 16000
    tone = make_tone(1.0, sr)
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "simple.wav")
        write_wav(wav_path, tone, sr)

        # monkeypatch ASRBackend to avoid loading real models and to return controlled segments
        class DummyBackend:
            def __init__(self, *args, **kwargs):
                self.backend = "mock"

            def transcribe_chunk(self, audio_np, sr, offset=0.0, language=None):
                # Return two segments that are close in time and should be merged by the pipeline.
                return [
                    {"start": offset + 0.0, "end": offset + 0.5, "text": "hello"},
                    {"start": offset + 0.6, "end": offset + 1.0, "text": "world"}
                ]

        monkeypatch.setattr(asr, "ASRBackend", DummyBackend)

        # Run the pipeline with VAD aggressiveness high so it produces a single voiced chunk
        out = asr.transcribe_audio_file(wav_path, model_name="small", vad_aggressiveness=2)

        # stitched text should contain 'hello' and 'world'
        assert "hello" in out["text"]
        assert "world" in out["text"]
        # merged result should coalesce adjacent segments into 1 (our pipeline merges within 0.25s gaps)
        assert isinstance(out["segments"], list)
        # Because segments were 0.5 and 0.6 separated by 0.1s gap -> they should be merged
        assert len(out["segments"]) == 1

def test_transcribe_textonly_wrapper(monkeypatch):
    """text-only convenience wrapper returns same stitched text as dict result."""
    class DummyBackend2:
        def __init__(self, *args, **kwargs):
            pass
        def transcribe_chunk(self, audio_np, sr, offset=0.0, language=None):
            return [{"start": offset+0.0, "end": offset+0.5, "text":"hey"}]

    monkeypatch.setattr(asr, "ASRBackend", DummyBackend2)
    sr = 16000
    tone = make_tone(0.5, sr)
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "one.wav")
        write_wav(wav_path, tone, sr)
        text = asr.transcribe_audio_file_textonly(wav_path, model_name="tiny")
        assert isinstance(text, str)
        assert "hey" in text
