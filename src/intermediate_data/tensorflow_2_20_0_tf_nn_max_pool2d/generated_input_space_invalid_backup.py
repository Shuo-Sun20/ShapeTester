import tensorflow as tf
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.random.normal(shape=(2, 8, 8, 3)),
    'ksize': (2, 2),
    'strides': (2, 2),
    'padding': 'VALID',
    'data_format': 'NHWC',
    'name': None
}

# 2. & 3. Identify shape-affecting parameters and their value spaces
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters of tf.nn.max_pool2d (except 'inputs')
    that affect output shape, with discretized value ranges.
    """
    ksize: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2), (2, 3)]
    )
    strides: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2), (2, 2)]
    )
    padding: List[Union[str, List[List[int]]]] = field(
        default_factory=lambda: [
            'VALID',
            'SAME',
            [[0, 0], [0, 1], [0, 1], [0, 0]],
            [[0, 0], [1, 0], [0, 0], [0, 0]],
            [[0, 0], [1, 1], [1, 1], [0, 0]]
        ]
    )
    data_format: List[str] = field(
        default_factory=lambda: ['NHWC', 'NCHW']
    )