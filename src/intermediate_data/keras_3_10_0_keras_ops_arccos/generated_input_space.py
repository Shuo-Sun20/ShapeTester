import keras
from dataclasses import dataclass
from typing import Any

valid_test_case = {
    "inputs": keras.ops.convert_to_tensor([0.5, -0.5, 0.0, 0.707, -0.707], dtype="float32")
}

@dataclass
class InputSpace:
    pass