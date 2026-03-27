import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    "inputs": np.random.random((2, 4, 4, 3)).astype(np.float32),
    "data_format": None
}

@dataclass
class InputSpace:
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )