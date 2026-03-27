import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# Original cell definition and call_func from the question
class MinimalRNNCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='uniform',
            name='kernel'
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel'
        )

    def call(self, inputs, states):
        prev_output = states[0]
        h = keras.ops.matmul(inputs, self.kernel)
        output = h + keras.ops.matmul(prev_output, self.recurrent_kernel)
        return output, [output]

def call_func(
    cell,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    zero_output_for_mask=False,
    inputs=None,
    initial_state=None,
    mask=None,
    training=False
):
    layer = keras.layers.RNN(
        cell=cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        zero_output_for_mask=zero_output_for_mask
    )
    return layer(inputs, initial_state=initial_state, mask=mask, training=training)

# Create some cell configurations for testing
cell_configs = [
    MinimalRNNCell(16),
    MinimalRNNCell(32),
    MinimalRNNCell(64),
    [MinimalRNNCell(16), MinimalRNNCell(32)],
    [MinimalRNNCell(32), MinimalRNNCell(64)],
    [MinimalRNNCell(16), MinimalRNNCell(32), MinimalRNNCell(64)]
]

valid_test_case = {
    'cell': cell_configs[1],  # MinimalRNNCell(32)
    'return_sequences': False,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
    'zero_output_for_mask': False,
    'inputs': np.random.randn(2, 10, 5).astype(np.float32),
    'initial_state': None,
    'mask': None,
    'training': False
}

@dataclass
class InputSpace:
    # Parameters that affect output shape (excluding 'inputs')
    cell: List[Union[MinimalRNNCell, List[MinimalRNNCell]]] = field(
        default_factory=lambda: cell_configs
    )
    return_sequences: List[bool] = field(
        default_factory=lambda: [False, True]
    )
    return_state: List[bool] = field(
        default_factory=lambda: [False, True]
    )