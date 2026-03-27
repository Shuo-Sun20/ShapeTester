import keras
from dataclasses import dataclass, field
from typing import Optional, Union, List
import numpy as np

# Generate tensors using keras.random module
query = keras.random.normal((2, 4, 8, 16))
key = keras.random.normal((2, 6, 8, 16))
value = keras.random.normal((2, 6, 8, 16))

valid_test_case = {
    "inputs": [query, key, value],
    "bias": None,
    "mask": None,
    "scale": None,
    "is_causal": False,
    "flash_attention": None,
    "attn_logits_soft_cap": None
}

@dataclass
class InputSpace:
    bias: List[Optional[np.ndarray]] = field(
        default_factory=lambda: [
            None,
            np.random.randn(1, 1, 4, 6),  # (1, 1, T, S)
            np.random.randn(2, 8, 4, 6),  # (B, N, T, S)
            np.random.randn(2, 1, 1, 6),  # (B, 1, 1, S)
            np.random.randn(1, 8, 1, 6),  # (1, N, 1, S)
        ]
    )
    
    mask: List[Optional[np.ndarray]] = field(
        default_factory=lambda: [
            None,
            np.array([[[[True]*6]]*4]*8).transpose(2,0,1,3),  # (1, N, T, S)
            np.array([[[[True]*6]]*4]*2).transpose(0,2,1,3),  # (B, 1, T, S)
            np.random.choice([True, False], size=(2, 8, 4, 6)),  # (B, N, T, S)
            np.array([[[[True]*6]]*4]).transpose(1,0,2,3),  # (N, 1, T, S)
        ]
    )
    
    scale: List[Optional[float]] = field(
        default_factory=lambda: [
            None,
            0.0,  # zero scale (edge case)
            0.125,  # 1/sqrt(64) for H=64
            0.25,  # 1/sqrt(16) for H=16 (default)
            0.5,  # 1/sqrt(4)
            1.0,  # unit scale
            2.0,  # double scale
            10.0,  # large scale (edge case)
            -0.25,  # negative scale (edge case)
        ]
    )
    
    is_causal: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    flash_attention: List[Optional[bool]] = field(
        default_factory=lambda: [True, False, None]
    )
    
    attn_logits_soft_cap: List[Optional[float]] = field(
        default_factory=lambda: [
            None,
            0.0,  # zero cap (edge case)
            1.0,  # small cap
            10.0,  # medium cap
            100.0,  # large cap
            float('inf'),  # infinite cap (edge case)
            -10.0,  # negative cap (edge case)
        ]
    )