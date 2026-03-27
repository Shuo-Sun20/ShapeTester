from dataclasses import dataclass
import numpy as np

# 1. Valid test case
valid_test_case = {"inputs": np.random.randn(3, 2)}

# 2-4. Parameter analysis and InputSpace definition
@dataclass
class InputSpace:
    """
    Contains all parameters affecting output shape for view_as_complex.
    The only parameter is 'inputs', but per instructions, we exclude it.
    Since there are no additional parameters in call_func's signature
    beyond 'inputs' that affect output shape, this class has no fields.
    """
    pass