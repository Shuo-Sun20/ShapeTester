import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Union, List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": (torch.randn(3, 2, 8), torch.randn(4, 2, 8), torch.randn(4, 2, 8)),
    "embed_dim_to_check": 8,
    "num_heads": 2,
    "in_proj_weight": torch.randn(24, 8),
    "in_proj_bias": torch.randn(24),
    "bias_k": None,
    "bias_v": None,
    "add_zero_attn": False,
    "dropout_p": 0.0,
    "out_proj_weight": torch.randn(8, 8),
    "out_proj_bias": torch.randn(8),
    "training": True,
    "key_padding_mask": None,
    "need_weights": True,
    "attn_mask": None,
    "is_causal": False,
    "use_separate_proj_weight": False,
    "q_proj_weight": None,
    "k_proj_weight": None,
    "v_proj_weight": None,
    "static_k": None,
    "static_v": None,
    "average_attn_weights": True
}

# 2 & 3. Parameters affecting output shape and their value spaces
# Based on analysis:
# - embed_dim_to_check (int): Must match query's last dimension and be divisible by num_heads
# - num_heads (int): Must divide embed_dim_to_check
# - use_separate_proj_weight (bool): Affects projection weight usage
# - q_proj_weight, k_proj_weight, v_proj_weight: When use_separate_proj_weight=True
# - static_k, static_v: Can override key/value sequences
# Note: input tensors (query, key, value) themselves affect shape but are excluded per requirements

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output tensor shape with discretized value ranges."""
    
    # Discrete parameters
    embed_dim_to_check: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    num_heads: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    use_separate_proj_weight: List[bool] = field(default_factory=lambda: [False, True])
    add_zero_attn: List[bool] = field(default_factory=lambda: [False, True])
    need_weights: List[bool] = field(default_factory=lambda: [False, True])
    is_causal: List[bool] = field(default_factory=lambda: [False, True])
    average_attn_weights: List[bool] = field(default_factory=lambda: [False, True])
    
    # Continuous parameters (discretized)
    dropout_p: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    
    # Note: The following tensor parameters are included for completeness but 
    # their shape constraints depend on other parameters
    # We include placeholder values indicating they can be None or tensors
    bias_k: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    bias_v: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    key_padding_mask: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    attn_mask: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    static_k: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    static_v: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    q_proj_weight: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    k_proj_weight: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])
    v_proj_weight: List[Optional[torch.Tensor]] = field(default_factory=lambda: [None])