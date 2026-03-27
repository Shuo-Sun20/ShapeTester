import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union

# Define tensors for valid test case
batch_size = 2
seq_len = 5
feature_dim = 8
query_tensor = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
value_tensor = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
key_tensor = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)

valid_test_case = {
    "inputs": [query_tensor, value_tensor],
    "num_heads": 2,
    "key_dim": 4,
    "value_dim": 4,
    "dropout": 0.0,
    "use_bias": True,
    "output_shape": None,
    "attention_axes": None,
    "flash_attention": None,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
    "seed": None,
    "attention_mask": None,
    "return_attention_scores": False,
    "training": False,
    "use_causal_mask": False
}

@dataclass
class InputSpace:
    """Dataclass containing parameters that affect output shape with discretized value ranges."""
    
    output_shape: list = field(
        default_factory=lambda: [None, 8, 16, 32, 64]
    )