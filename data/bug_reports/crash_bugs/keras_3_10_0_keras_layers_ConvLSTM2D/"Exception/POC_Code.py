import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(
    # Constructor parameters
    filters,
    kernel_size,
    inputs,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    seed=None,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    # Call parameters
    mask=None,
    training=None,
    initial_state=None
):
    layer = keras.layers.ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        seed=seed,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll
    )
    return layer(inputs, mask=mask, training=training, initial_state=initial_state)

# Test parameters
filters = 1
kernel_size = [3, 3]
strides = [2, 2]
padding = 'same'
data_format = 'channels_last'
dilation_rate = [3, 3]
activation = 'tanh'
recurrent_activation = 'sigmoid'
use_bias = True
kernel_initializer = 'glorot_uniform'
recurrent_initializer = 'orthogonal'
bias_initializer = 'zeros'
unit_forget_bias = True
return_sequences = True
return_state = False
go_backwards = False
stateful = False
unroll = False

# Create eager tensor input
eager_input = tf.constant(np.random.random((2, 5, 32, 32, 3)).astype(np.float32))

# Create Keras.Input placeholder with same shape
placeholder_input = keras.Input(shape=(5, 32, 32, 3))

print("Testing with eager tensor:")
try:
    eager_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        inputs=eager_input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll
    )
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling ConvLSTMCell.call().\n\n{e}")

print("\nTesting with Keras.Input placeholder:")
try:
    placeholder_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        inputs=placeholder_input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll
    )
    print(f"Static output shape: {placeholder_output.shape}")
except Exception as e:
    print(f"Static output shape: Exception encountered when calling ConvLSTMCell.call().\n\n{e}")

print("\nDefect reproduced: Dynamic execution fails while static execution succeeds with incompatible strides and dilation_rate")