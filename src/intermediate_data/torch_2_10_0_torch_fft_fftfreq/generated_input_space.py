import torch
from dataclasses import dataclass, field
from typing import Optional, List, Any

# 1. Define valid_test_case
valid_test_case = {
    'n': 5,
    'd': 1.0,
    'out': None,
    'dtype': None,
    'layout': torch.strided,
    'device': None,
    'requires_grad': False,
    'inputs': None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    n: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 10, 100, 1000, 10000])
    # Note: d does not affect output shape, so it's not included in InputSpace

# 2. Analysis of parameters affecting output shape:
# Only 'n' affects the shape of the output tensor, as it determines the length
# of the frequency array. The other parameters affect dtype, device, layout,
# or gradient tracking but not the shape.

# 3. Parameter value space analysis:
# - n: Discrete positive integer parameter
#   Value space includes:
#   * Boundary values: 1 (smallest positive integer)
#   * Typical values: 2, 3, 4, 5, 10, 100, 1000, 10000
#   * Note: 5 is included from valid_test_case
#
# - d: Continuous positive float parameter (does not affect shape)
# - out: Tensor or None (does not affect shape when None)
# - dtype: torch.dtype or None (does not affect shape)
# - layout: torch.layout (discrete, does not affect shape)
# - device: torch.device or None (does not affect shape)
# - requires_grad: bool (discrete, does not affect shape)
# - inputs: Not used in this API, always None

# The InputSpace class can be instantiated as var=InputSpace()