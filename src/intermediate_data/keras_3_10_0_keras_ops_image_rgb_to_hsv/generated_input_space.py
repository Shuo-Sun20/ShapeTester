import numpy as np
from dataclasses import dataclass, field

# Valid test case as required
valid_test_case = {
    'inputs': np.random.random((4, 4, 3)).astype(np.float32),
    'data_format': None
}

@dataclass
class InputSpace:
    """Class containing parameters that affect output tensor shape."""
    data_format: list = field(default_factory=lambda: [
        None,  # default (channels_last)
        "channels_last",
        "channels_first"
    ])