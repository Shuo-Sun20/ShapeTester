import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Define the tensors for the valid test case
np.random.seed(42)
real_tensor = keras.ops.convert_to_tensor(np.random.randn(8).astype(np.float32))
imag_tensor = keras.ops.convert_to_tensor(np.random.randn(8).astype(np.float32))

valid_test_case = {
    "inputs": [real_tensor, imag_tensor],
    "fft_length": 16
}

@dataclass
class InputSpace:
    """Parameter space for testing keras.ops.irfft."""
    
    fft_length: List[Optional[int]] = field(
        default_factory=lambda: [
            None,  # inferred from input
            1,     # minimum valid length
            2,     # small even length
            3,     # odd length (requires explicit specification)
            4,     # even length
            8,     # even length (matches example)
            10,    # even length
            15,    # odd length
            16,    # power of two (matches test case)
            32,    # larger power of two
            50,    # larger even length
            101    # larger odd length
        ]
    )