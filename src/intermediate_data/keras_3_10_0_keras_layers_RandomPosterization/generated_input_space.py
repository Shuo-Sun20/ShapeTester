from dataclasses import dataclass
import numpy as np
import keras

# Task 1: Define valid_test_case
valid_test_case = {
    "factor": 4,
    "inputs": np.random.uniform(0, 255, size=(2, 32, 32, 3)).astype(np.float32),
    "value_range": (0, 255),
    "data_format": None,
    "seed": None
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Task 2: Identify shape-affecting parameters (besides "inputs")
    # - `data_format` can change axis ordering, affecting shape interpretation
    # - `factor`, `value_range`, and `seed` do NOT affect tensor shape
    data_format: list = None
    
    def __post_init__(self):
        # Task 3: Value space construction
        # Discrete parameter: data_format has limited possible values
        if self.data_format is None:
            # All possible values per Keras conventions
            self.data_format = [None, "channels_last", "channels_first"]