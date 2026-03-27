import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    "negative_slope": 0.2,
    "inputs": tf.random.uniform(shape=(2, 5), minval=-1.0, maxval=1.0),
    "name": "leaky_relu_example",
    "dtype": "float32"
}

@dataclass
class InputSpace:
    negative_slope: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    )
    name: List[Optional[str]] = field(
        default_factory=lambda: [None, "leaky_relu_example", "custom_name_1", "custom_name_2"]
    )
    dtype: List[Optional[str]] = field(
        default_factory=lambda: [None, "float32", "float64", "bfloat16", "float16"]
    )