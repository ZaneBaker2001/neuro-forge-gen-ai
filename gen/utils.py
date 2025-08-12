import os
import hashlib
import random
from dataclasses import dataclass
from typing import Optional, List

try:
    import torch
except Exception:
    torch = None

SEED_ENV = "GENAI_STUDIO_SEED"

def set_seed(seed: Optional[int] = None) -> int:
    """
    Set global seed for reproducibility. Returns the seed actually used.
    """
    if seed is None:
        env_seed = os.getenv(SEED_ENV)
        if env_seed:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = random.randint(0, 2**32 - 1)
        else:
            seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    return seed

def device_str() -> str:
    if torch is None:
        return "cpu (torch not available)"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]

@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_new_tokens: int = 256
    top_p: float = 0.95
    top_k: int = 50

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
    """
    Simple overlap-based text chunker.
    """
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        j = min(i + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[i:j]))
        i = j - overlap
        if i < 0:
            i = 0
        if i >= len(tokens):
            break
    return chunks
