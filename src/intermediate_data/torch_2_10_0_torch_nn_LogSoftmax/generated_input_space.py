import torch
from dataclasses import dataclass, field

valid_test_case = {
    'dim': 1,
    'inputs': torch.randn(2, 3)
}

@dataclass
class InputSpace:
    dim: list = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2, 3])
    # Note: inputs is not included here because:
    # 1. It's excluded by task requirement (except for "inputs")
    # 2. Its value space would be infinite (any tensor shape)
    # Only 'dim' parameter affects computation along a dimension
    
    # No other parameters in call_func() affect output shape
    # Output shape always equals input shape regardless of 'dim' value