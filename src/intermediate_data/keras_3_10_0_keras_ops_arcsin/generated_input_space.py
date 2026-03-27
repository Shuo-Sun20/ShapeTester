import keras
import keras.ops as ops
from dataclasses import dataclass
from typing import List, Union

valid_test_case = {
    'inputs': keras.random.uniform(shape=(3,), minval=-1.0, maxval=1.0)
}

@dataclass
class InputSpace:
    pass