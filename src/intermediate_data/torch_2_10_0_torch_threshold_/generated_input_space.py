import torch
from dataclasses import dataclass

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(3, 4)],
    'threshold': 0.5,
    'value': 0.0
}

# Task 2: Identify parameters affecting output shape (excluding 'inputs')
# The torch.threshold_ API only applies element-wise operations. 
# The output shape is identical to the input tensor shape. 
# Therefore, 'threshold' and 'value' do NOT affect the output shape.
# Only 'inputs' (specifically the shape of the tensor in inputs[0]) affects output shape.
# Since we exclude 'inputs', there are no parameters left that affect shape.

# Task 3 & 4: Define InputSpace class with all shape-affecting parameters (none in this case)
@dataclass
class InputSpace:
    pass