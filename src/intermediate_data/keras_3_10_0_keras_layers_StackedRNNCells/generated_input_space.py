import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
batch_size = 3
features = 5
units = 128

input_tensor = np.random.randn(batch_size, features).astype(np.float32)
rnn_cells = [keras.layers.LSTMCell(units) for _ in range(2)]
h_state_1 = np.random.randn(batch_size, units).astype(np.float32)
c_state_1 = np.random.randn(batch_size, units).astype(np.float32)
h_state_2 = np.random.randn(batch_size, units).astype(np.float32)
c_state_2 = np.random.randn(batch_size, units).astype(np.float32)

inputs_list = [input_tensor, h_state_1, c_state_1, h_state_2, c_state_2]

valid_test_case = {
    'cells': rnn_cells,
    'inputs': inputs_list,
    'training': False
}

# 2-4. Define InputSpace dataclass
@dataclass
class InputSpace:
    cells: List[List[Union[keras.layers.LSTMCell, keras.layers.GRUCell, keras.layers.SimpleRNNCell]]] = field(
        default_factory=lambda: [
            [keras.layers.LSTMCell(128)],
            [keras.layers.LSTMCell(128), keras.layers.LSTMCell(128)],
            [keras.layers.GRUCell(64)],
            [keras.layers.GRUCell(64), keras.layers.GRUCell(64)],
            [keras.layers.SimpleRNNCell(32)]
        ]
    )