import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(
    filters,
    kernel_size,
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
    inputs=None,
    mask=None,
    training=None,
    initial_state=None
):
    layer = keras.layers.ConvLSTM3D(
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
    output = layer(inputs, mask=mask, training=training, initial_state=initial_state)
    return output

# Test parameters
filters = 4
kernel_size = [3, 3, 5]
strides = [2, 2, 2]
padding = 'same'
data_format = 'channels_first'
dilation_rate = [2, 2, 2]

# Create eager tensor input
eager_input = tf.random.normal((2, 5, 8, 8, 8, 3))

# Create Keras Input placeholder with same shape
placeholder_input = keras.Input(shape=(5, 8, 8, 8, 3))

print("Testing with eager tensor:")
try:
    eager_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        inputs=eager_input
    )
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered - {e}")

print("\nTesting with Keras Input placeholder:")
try:
    placeholder_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        inputs=placeholder_input
    )
    print(f"Static output shape: {placeholder_output.shape}")
except Exception as e:
    print(f"Static output shape: Exception encountered - {e}")