import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List
import math

# 1. Define valid_test_case
batch_size = 2
num_heads_query = 8
num_heads_kv = 4
seq_len_q = 10
seq_len_kv = 12
embed_dim = 64
value_dim = 48

query = torch.randn(batch_size, num_heads_query, seq_len_q, embed_dim)
key = torch.randn(batch_size, num_heads_kv, seq_len_kv, embed_dim)
value = torch.randn(batch_size, num_heads_kv, seq_len_kv, value_dim)

valid_test_case = {
    "inputs": [query, key, value],
    "attn_mask": None,
    "dropout_p": 0.1,
    "is_causal": True,
    "scale": None,
    "enable_gqa": True
}

# 2. & 3. Parameters affecting output shape and their value spaces
# Only `enable_gqa` directly affects shape transformation through internal repetition

@dataclass
class InputSpace:
    # enable_gqa: Boolean parameter affecting internal key/value repetition
    enable_gqa: List[bool] = field(default_factory=lambda: [False, True])
    
    # attn_mask: Can affect computation but not output shape
    attn_mask: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool),
        torch.zeros(seq_len_q, seq_len_kv, dtype=torch.bool),
        torch.triu(torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool), diagonal=1),
        torch.randn(seq_len_q, seq_len_kv, dtype=torch.float32) * 0.1,
    ])
    
    # dropout_p: Continuous parameter discretized
    dropout_p: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0
    ])
    
    # is_causal: Discrete boolean parameter
    is_causal: List[bool] = field(default_factory=lambda: [False, True])
    
    # scale: Optional float parameter
    scale: List[Optional[float]] = field(default_factory=lambda: [
        None,
        0.5,
        1.0,
        1.5,
        2.0,
        1/math.sqrt(embed_dim)  # Default scaling factor
    ])