import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
example_input = torch.randn(3, 4)
valid_test_case = {
    'inputs': example_input,
    'lower': 1./8,
    'upper': 1./3,
    'training': True,
    'inplace': False
}

# 2. Parameters affecting output shape (except "inputs"): None
# F.rrelu output shape is solely determined by input shape, other parameters only affect values.

# 3. Value spaces for all parameters (except inputs)
# lower: continuous float [0, inf), typical for RReLU is [0, 1)
lower_values = [0.0, 0.01, 0.125, 0.25, 0.5, 0.99]

# upper: continuous float [0, inf), should be >= lower
upper_values = [0.01, 0.125, 0.333, 0.5, 1.0, 2.0]

# training: discrete boolean
training_values = [True, False]

# inplace: discrete boolean
inplace_values = [True, False]

# 4. Define InputSpace dataclass (empty since no shape-affecting parameters except inputs)
@dataclass
class InputSpace:
    """
    Contains all parameters that affect output tensor shape (except inputs).
    Since F.rrelu output shape is determined only by input shape,
    this dataclass is empty but defined for interface consistency.
    """
    pass