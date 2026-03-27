import keras
import numpy as np
from dataclasses import dataclass, field

# 1. Define valid_test_case variable
np.random.seed(42)
indices_tensor = keras.ops.convert_to_tensor(
    np.random.randint(0, 12, size=(4,)), dtype="int32"
)
shape_tuple = (3, 4)
valid_test_case = {
    "inputs": indices_tensor,
    "shape": shape_tuple
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only parameter affecting output shape (excluding "inputs") is "shape"
    # shape: tuple or list defining array dimensions
    shape: list = field(default_factory=lambda: [
        # Discrete values covering all legal scenarios:
        # 1. Single dimension shapes
        (1,), (2,), (5,), (10,), (100,),
        # 2. 2D shapes
        (1, 1), (1, 5), (5, 1), (3, 4), (10, 10),
        # 3. 3D shapes
        (1, 1, 1), (2, 3, 4), (5, 5, 5),
        # 4. Higher dimensional shapes
        (2, 2, 2, 2), (3, 1, 4, 1, 5),
        # 5. Boundary cases (minimum and maximum practical dimensions)
        (),  # 0-dimensional (scalar array)
        (2, 3, 4, 5, 6)  # 5-dimensional
    ])