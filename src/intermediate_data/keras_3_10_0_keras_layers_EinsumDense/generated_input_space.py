import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

# 1. Valid test case
valid_test_case = {
    "equation": "...x,xy->...y",
    "output_shape": 64,
    "activation": None,
    "bias_axes": "y",
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
    "lora_rank": None,
    "lora_alpha": None,
    "inputs": np.random.randn(5, 32, 128).astype(np.float32)
}

# 2. Parameters affecting output shape: equation, output_shape
# 3. Discretized value spaces

@dataclass
class InputSpace:
    # equation: 5 representative einsum patterns covering common use cases
    equation: List[str] = field(default_factory=lambda: [
        "ab,bc->ac",           # Standard dense layer
        "abc,cd->abd",         # Dense applied to sequence (no ellipsis)
        "...x,xy->...y",       # Dense with ellipsis (single feature dim)
        "...ab,bc->...ac",     # Dense with ellipsis (multiple preserved dims)
        "abcd,de->abce"        # Dense applied to 3D+ data
    ])
    
    # output_shape: 5 representative shapes covering different dimensionalities
    output_shape: List[Union[int, Tuple[Optional[int], ...]]] = field(default_factory=lambda: [
        64,                    # Single output dimension
        (None, 64),           # 2D output with one inferred dimension
        (None, None, 32),     # 3D output with two inferred dimensions
        (128,),               # Single dimension (explicit)
        (None, 64, 32)        # 3D output with one inferred dimension
    ])