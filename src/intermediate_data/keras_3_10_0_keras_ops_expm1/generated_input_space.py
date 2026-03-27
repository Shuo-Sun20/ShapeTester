import keras
import numpy as np
from dataclasses import dataclass

keras.utils.set_random_seed(42)
valid_test_case = {'inputs': keras.random.normal(shape=(3, 4))}

@dataclass
class InputSpace:
    # No additional parameters affect output shape beyond 'inputs' (which is excluded)
    pass