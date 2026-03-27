import torch
from dataclasses import dataclass
from typing import Optional, List, Union

def call_func(inputs, out=None):
    return torch.special.i1e(inputs, out=out)

# 1. Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(3, 4),
    'out': None
}

# 2. Parameters affecting output shape (except "inputs"): only "out"
# 3. Value space for "out": discretized to include None and tensors of different shapes
#    Since "out" can be None or a tensor, and when provided, its shape must match the input's shape.
#    However, to cover all legal scenarios, we include None and tensors of various shapes that could be used in valid calls.
#    Note: For invalid shapes, the function would raise an error, so we only consider valid ones.
#    We include None and a few tensor shapes that are common.

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # The "out" parameter can affect the output shape when provided, as the output will be the provided tensor.
    # We discretize the value space for "out" to include None and several tensor shapes.
    # Since "out" must have the same shape as the input in a valid call, we only include shapes that match the input.
    # However, for the purpose of this class, we list possible values for "out" as examples.
    # We include None and a few example tensors of different shapes and dtypes.
    out: List[Optional[torch.Tensor]] = None

    def __post_init__(self):
        if self.out is None:
            # Define a list of discretized values for "out"
            # Include None and tensors of various shapes and dtypes that are valid for typical inputs.
            self.out = [
                None,
                torch.empty(3, 4, dtype=torch.float32),
                torch.empty(3, 4, dtype=torch.float64),
                torch.empty(1, 12, dtype=torch.float32),  # Same number of elements, different shape
                torch.empty(6, 2, dtype=torch.float32),   # Same number of elements, different shape
                torch.empty(2, 6, dtype=torch.float32),   # Same number of elements, different shape
            ]