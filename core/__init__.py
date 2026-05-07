import ctypes
import os
import numpy as np

_lib_dir = os.path.dirname(os.path.abspath(__file__))
_ext = ".dll" if os.name == "nt" else ".so"
_lib_path = os.path.join(_lib_dir, f"sampler{_ext}")
_lib = ctypes.CDLL(os.path.abspath(_lib_path), winmode=0)

_lib.sample_token.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # logits
    ctypes.c_int,                     # vocab_size
    ctypes.c_float,                   # temperature
    ctypes.c_int,                     # top_k
    ctypes.c_float,                   # top_p
    ctypes.c_uint,                    # seed
]
_lib.sample_token.restype = ctypes.c_int


def sample(logits: np.ndarray, temperature=0.7, top_k=50, top_p=0.9, seed=42) -> int:
    """Sample next token from logits using C++ sampler."""
    logits = logits.astype(np.float32).copy()
    ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return _lib.sample_token(ptr, len(logits), temperature, top_k, top_p, seed)