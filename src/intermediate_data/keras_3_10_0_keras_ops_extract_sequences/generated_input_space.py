import keras
from dataclasses import dataclass
from typing import List

def call_func(inputs, sequence_length, sequence_stride):
    return keras.ops.extract_sequences(inputs, sequence_length, sequence_stride)

# Task 1: Define valid_test_case
x = keras.random.uniform(shape=(100,))
valid_test_case = {
    "inputs": x,
    "sequence_length": 5,
    "sequence_stride": 2
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    sequence_length: List[int] = None
    sequence_stride: List[int] = None
    
    def __post_init__(self):
        if self.sequence_length is None:
            # Typical values covering boundaries and valid scenarios
            # Include 5 from valid_test_case
            self.sequence_length = [1, 3, 5, 10, 50, 99, 100]
        if self.sequence_stride is None:
            # Typical values covering boundaries and valid scenarios
            # Include 2 from valid_test_case
            self.sequence_stride = [1, 2, 3, 10, 25, 50, 100]