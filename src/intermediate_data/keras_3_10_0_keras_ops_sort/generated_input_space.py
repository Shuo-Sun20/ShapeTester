import keras
from dataclasses import dataclass

valid_test_case = {
    "inputs": keras.random.uniform(shape=(3, 4, 5)),
    "axis": -1
}

@dataclass
class InputSpace:
    axis: list = None
    
    def __post_init__(self):
        if self.axis is None:
            # For a 3D tensor with shape (3, 4, 5), valid axis values are:
            # -3, -2, -1 (negative indexing)
            # 0, 1, 2 (positive indexing)
            # None (flatten before sorting)
            self.axis = [None, -3, -2, -1, 0, 1, 2]