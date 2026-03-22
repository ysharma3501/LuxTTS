import pytest
import torch
import numpy as np
import soundfile as sf
from zipvoice.luxvoice import LuxTTS

def test_fp16_loads_without_error(lux16):
    assert lux16 is not None

def test_fp16_model_dtype(lux16):
    if lux16.device == "cpu":
        assert lux16.dtype == torch.float32
    else:
        param = next(lux16.model.parameters())
        assert param.dtype == torch.float16, f"Expected float16, got {param.dtype}"

def test_fp32_model_dtype(lux32):
    if lux32.device == "cpu":
        assert lux32.dtype == torch.float32
    else:
        param = next(lux32.model.parameters())
        assert param.dtype == torch.float32, f"Expected float32, got {param.dtype}"

def test_fp16_output_is_float32(lux16, reference_audio, test_text):
    enc = lux16.encode_prompt(reference_audio, rms=0.01)
    wav = lux16.generate_speech(test_text, enc, num_steps=4)
    assert wav.dtype == torch.float32

def test_fp16_no_nan_in_output(lux16, reference_audio, test_text):
    enc = lux16.encode_prompt(reference_audio, rms=0.01)
    wav = lux16.generate_speech(test_text, enc, num_steps=4)
    assert not wav.isnan().any(), "NaN detected in fp16 output"

def test_fp16_no_inf_in_output(lux16, reference_audio, test_text):
    enc = lux16.encode_prompt(reference_audio, rms=0.01)
    wav = lux16.generate_speech(test_text, enc, num_steps=4)
    assert not wav.isinf().any(), "Inf detected in fp16 output"

def test_fp16_output_in_valid_range(lux16, reference_audio, test_text):
    enc = lux16.encode_prompt(reference_audio, rms=0.01)
    wav = lux16.generate_speech(test_text, enc, num_steps=4).numpy()
    assert wav.max() <= 1.0 and wav.min() >= -1.0, "Waveform outside [-1, 1]"

def test_fp16_output_is_not_silent(lux16, reference_audio, test_text):
    enc = lux16.encode_prompt(reference_audio, rms=0.01)
    wav = lux16.generate_speech(test_text, enc, num_steps=4).numpy()
    assert np.abs(wav).mean() > 1e-4, "Output waveform is silent"

def test_default_dtype_is_float32(lux32):
    assert lux32.dtype == torch.float32

def test_fp32_no_nan_in_output(lux32, reference_audio, test_text):
    enc = lux32.encode_prompt(reference_audio, rms=0.01)
    wav = lux32.generate_speech(test_text, enc, num_steps=4)
    assert not wav.isnan().any()

def test_fp32_output_is_float32(lux32, reference_audio, test_text):
    enc = lux32.encode_prompt(reference_audio, rms=0.01)
    wav = lux32.generate_speech(test_text, enc, num_steps=4)
    assert wav.dtype == torch.float32

def test_fp16_falls_back_on_cpu(capsys):
    lux = LuxTTS("YatharthS/LuxTTS", device="cpu", dtype="float16")
    assert lux.dtype == torch.float32
    captured = capsys.readouterr()
    assert "float32" in captured.out
