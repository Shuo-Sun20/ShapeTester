import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

# Task 1: Define valid_test_case variable
valid_test_case = {
    'inputs': [
        tf.random.uniform(shape=[100, 16], minval=-1.0, maxval=1.0),
        tf.SparseTensor(
            indices=[[0, 0], [0, 1], [1, 0], [2, 3], [2, 1]],
            values=[5, -1, 30, 15, 7],
            dense_shape=[3, 4]
        ),
        tf.SparseTensor(
            indices=[[0, 0], [0, 1], [1, 0], [2, 3], [2, 1]],
            values=[1.5, 0.0, 0.8, 2.0, 1.2],
            dense_shape=[3, 4]
        )
    ],
    'combiner': 'mean',
    'default_id': None,
    'max_norm': 1.0,
    'allow_fast_lookup': True
}

# Task 2 & 3: Identify shape-affecting parameters and their value spaces
# The parameters are combiner and max_norm
# combiner: discrete with 3 values
# max_norm: continuous parameter, discretize to 5 values (including None)

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # combiner: discrete parameter, all possible values
    combiner: list[str] = field(default_factory=lambda: ["mean", "sum", "sqrtn"])
    # max_norm: continuous parameter, discretized to 5 values
    max_norm: list[float | None] = field(default_factory=lambda: [None, 0.1, 1.0, 5.0, 10.0])