import pytest
from zipvoice.luxvoice import LuxTTS


@pytest.fixture(scope="session")
def lux32():
    return LuxTTS("YatharthS/LuxTTS", device="cuda", dtype="float32")


@pytest.fixture(scope="session")
def lux16():
    return LuxTTS("YatharthS/LuxTTS", device="cuda", dtype="float16")


@pytest.fixture(scope="session")
def reference_audio():
    return "tests/assets/reference.wav"


@pytest.fixture(scope="session")
def test_text():
    return "The quick brown fox jumps over the lazy dog."
