import keras
import numpy as np

def call_func(
    units,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    seed=None,
    inputs=None,
    training=False
):
    sequence, states = inputs[0], inputs[1]
    cell = keras.layers.SimpleRNNCell(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        seed=seed
    )
    output, next_states = cell(sequence, states, training=training)
    return output

sequence = np.random.random((32, 8)).astype(np.float32)
states = np.random.random((32, 4)).astype(np.float32)
example_output = call_func(
    units=4,
    inputs=[sequence, states],
    training=False
)