import tensorflow as tf
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# 1. Define valid_test_case
example_input = tf.constant([[1, 2, 3], [4, 5, 6]])
valid_test_case = {
    "inputs": example_input,
    "name": None,
    "k": 1,
    "num_rows": 4,
    "num_cols": 4,
    "padding_value": 0,
    "align": "RIGHT_LEFT"
}

# 2. Parameters that affect output shape (excluding 'inputs'): k, num_rows, num_cols

# 3. Discretized value spaces for shape-affecting parameters
k_vals = [0, 1, -1, (-1, 0), (-1, 1)]  # 5 representative values
num_rows_vals = [None, 1, 3, 5, 10]    # 5 values including None
num_cols_vals = [None, 1, 3, 5, 10]    # 5 values including None

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    k: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: k_vals)
    num_rows: List[Optional[int]] = field(default_factory=lambda: num_rows_vals)
    num_cols: List[Optional[int]] = field(default_factory=lambda: num_cols_vals)