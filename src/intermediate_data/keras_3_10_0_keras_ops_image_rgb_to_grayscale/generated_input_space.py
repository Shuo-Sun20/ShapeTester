import numpy as np
from dataclasses import dataclass, field
from typing import Optional

valid_test_case = {
    "inputs": np.random.random((2, 4, 4, 3)),
    "data_format": None
}

@dataclass
class InputSpace:
    data_format: list[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )