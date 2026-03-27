import keras
import numpy as np

class MinimalRNNCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')

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

cell = MinimalRNNCell(32)
batch_size = 2
timesteps = 10
features = 5
inputs = np.random.randn(batch_size, timesteps, features).astype(np.float32)
example_output = call_func(cell=cell, inputs=inputs)