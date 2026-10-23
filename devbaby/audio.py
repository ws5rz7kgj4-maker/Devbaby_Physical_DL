"""Procedural speech (eSpeak NG formants) and analytic impact sounds.

No neural TTS, speech recognition, word tokenizer or pretrained audio model.
The synthesizer is single-threaded; use separate processes for parallel worlds.
"""
import ctypes as C
from functools import lru_cache
import hashlib
from pathlib import Path
import threading
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft

_LOCK = threading.RLock()
_ENGINE = None

class _Engine:
    def __init__(self):
        import espeakng_loader
        self.lib = C.CDLL(espeakng_loader.get_library_path())
        self.lib.espeak_Initialize.argtypes = [C.c_int, C.c_int, C.c_char_p, C.c_int]
        self.lib.espeak_Initialize.restype = C.c_int
        self.rate = self.lib.espeak_Initialize(2, 0, espeakng_loader.get_data_path().encode(), 0)
        if self.rate <= 0:
            raise RuntimeError("eSpeak NG initialization failed; do not replace speech with beeps.")
        callback = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_short), C.c_int, C.c_void_p)
        self.parts = []
        def collect(wav, n, events):
            if wav and n:
                self.parts.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
            return 0
        self.callback = callback(collect)
        self.lib.espeak_SetSynthCallback(self.callback)
        self.lib.espeak_SetVoiceByName.argtypes = [C.c_char_p]
        self.lib.espeak_SetVoiceByName.restype = C.c_int
        self.lib.espeak_SetParameter.argtypes = [C.c_int, C.c_int, C.c_int]
        self.lib.espeak_Synth.argtypes = [C.c_void_p, C.c_size_t, C.c_uint, C.c_int,
                                         C.c_uint, C.c_uint, C.c_void_p, C.c_void_p]
        self.lib.espeak_Synth.restype = C.c_int

    def speak(self, text, voice, speed, pitch):
        if self.lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise RuntimeError(f"Synthetic voice unavailable: {voice}")
        self.lib.espeak_SetParameter(1, int(speed), 0)
        self.lib.espeak_SetParameter(2, 80, 0)
        self.lib.espeak_SetParameter(3, int(pitch), 0)
        self.parts = []
        data = C.create_string_buffer(text.encode("utf-8") + b"\0")
        rc = self.lib.espeak_Synth(data, len(data), 0, 1, 0, 1, None, None)
        self.lib.espeak_Synchronize()
        if rc != 0 or not self.parts:
            raise RuntimeError("Speech synthesis produced no audio")
        return np.concatenate(self.parts).astype(np.float32) / 32768.0

class Speech:
    def __init__(self, cache, sample_rate=16000):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

    def synthesize(self, text, voice="en-us", speed=195, pitch=50):
        global _ENGINE
        key = hashlib.sha256(f"espeak-ng/{text}/{voice}/{speed}/{pitch}/{self.sample_rate}".encode()).hexdigest()
        path = self.cache / (key + ".wav")
        with _LOCK:
            if path.exists():
                rate, pcm = wavfile.read(path)
                assert rate == self.sample_rate
                return pcm.astype(np.float32) / 32768.0
            if _ENGINE is None:
                _ENGINE = _Engine()
            wav = _ENGINE.speak(text, voice, speed, pitch)
            import math
            d = math.gcd(self.sample_rate, _ENGINE.rate)
            wav = resample_poly(wav, self.sample_rate // d, _ENGINE.rate // d).astype(np.float32)
            # Keep amplitude fixed; no learned normalization or human recordings.
            pcm = np.clip(wav * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(path, self.sample_rate, pcm)
            return pcm.astype(np.float32) / 32768.0

@lru_cache(maxsize=8)
def mel_filters(sr=16000):
    hz_to_mel = lambda f: 2595 * np.log10(1 + np.asarray(f) / 700)
    mel_to_hz = lambda m: 700 * (10 ** (np.asarray(m) / 2595) - 1)
    points = mel_to_hz(np.linspace(hz_to_mel(60), hz_to_mel(sr / 2), 34))
    bins = np.linspace(0, sr / 2, 257)
    filters = []
    for a, b, c in zip(points[:-2], points[1:-1], points[2:]):
        w = np.maximum(0, np.minimum((bins-a)/(b-a), (c-bins)/(c-b)))
        filters.append(w / max(w.sum(), 1e-8))
    return np.asarray(filters, dtype=np.float32)

def features(wav, sr=16000):
    """Fixed 32x16 log-mel representation of one *past* control interval."""
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < 256:
        wav = np.pad(wav, (0, 256-len(wav)))
    _, _, s = stft(wav, fs=sr, nperseg=256, noverlap=128, nfft=512, boundary=None)
    x = np.log1p(1000 * (mel_filters(sr) @ np.abs(s))) / 6
    if x.shape[1] != 16:
        # Interpolation is within a fully observed interval, never into the future.
        old = np.linspace(0, 1, x.shape[1])
        x = np.stack([np.interp(np.linspace(0, 1, 16), old, row) for row in x])
    return x.astype(np.float32)

def query_features(wav, hz=8, sr=16000):
    n = sr // hz
    chunks = max(1, int(np.ceil(len(wav) / n)))
    wav = np.pad(wav, (0, chunks*n-len(wav)))
    return np.stack([features(wav[i*n:(i+1)*n], sr) for i in range(chunks)])

def impact(material, intensity, sr, rng):
    base, decay = {"wood": (420, 24), "metal": (1250, 7), "rubber": (150, 36)}[material]
    t = np.arange(int(sr * .35), dtype=np.float32) / sr
    signal = sum(np.sin(2*np.pi*base*f*t + rng.uniform(-.1, .1)) * np.exp(-decay*(1+i*.6)*t)/(i+1)
                 for i, f in enumerate([1, 1.61, 2.37]))
    return (signal * min(.25, .015*float(intensity))).astype(np.float32)
