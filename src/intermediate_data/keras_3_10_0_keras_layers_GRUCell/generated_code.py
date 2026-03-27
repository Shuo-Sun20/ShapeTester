import keras
import numpy as np

def call_func(
    units,
    activation="tanh",
    recurrent_activation="sigmoid",
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
    reset_after=True,
    seed=None,
    inputs=None,
    states=None,
    training=None
):
    gru_cell = keras.layers.GRUCell(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
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
        reset_after=reset_after,
        seed=seed
    )
    
    if states is not None:
        output, _ = gru_cell(inputs, states=states, training=training)
    else:
        output, _ = gru_cell(inputs, training=training)
    
    return output


batch_size = 32
features = 10
units = 4

input_tensor = np.random.random((batch_size, features))
state_tensor = np.random.random((batch_size, units))

example_output = call_func(
    units=units,
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    dropout=0.0,
    recurrent_dropout=0.0,
    reset_after=True,
    inputs=input_tensor,
    states=state_tensor,
    training=False
)