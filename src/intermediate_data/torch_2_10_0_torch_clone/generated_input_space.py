import torch
from dataclasses import dataclass

# 1. Valid test case
valid_test_case = {
    "inputs": [torch.randn(2, 3)],  # Must be list of tensors
    "memory_format": torch.preserve_format
}

# 2. Parameters affecting output shape (except 'inputs'): NONE
# torch.clone preserves input tensor shape regardless of memory_format

# 3. Value space analysis for memory_format parameter
# memory_format is a discrete parameter with 5 possible values

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # Only 'inputs' affects shape, but excluded per requirements.
    # memory_format doesn't affect shape but is included for completeness
    memory_format: list = None
    
    def __post_init__(self):
        if self.memory_format is None:
            # Discrete value space for memory_format parameter
            self.memory_format = [
                torch.preserve_format,
                torch.contiguous_format,
                torch.channels_last,
                torch.channels_last_3d,
                torch.channels_last_3d
            ]