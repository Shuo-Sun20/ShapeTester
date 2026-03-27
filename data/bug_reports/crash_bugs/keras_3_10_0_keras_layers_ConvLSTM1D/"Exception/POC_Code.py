import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
import numpy as np

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
    initial_state=None,
    mask=None,
    training=False
):
    layer = keras.layers.ConvLSTM1D(
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
    return layer(inputs, initial_state=initial_state, mask=mask, training=training)

# Test parameters from the defect description
filters = 1
kernel_size = 1
strides = 2
padding = 'same'
data_format = 'channels_last'
dilation_rate = 3
return_sequences = True

# Create eager tensor input
eager_input = tf.constant(np.random.random((4, 10, 32, 3)), dtype=tf.float32)

# Create Keras.Input placeholder with same shape
placeholder_input = keras.Input(shape=(10, 32, 3))

print("Testing with eager tensor input:")
try:
    dynamic_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        return_sequences=return_sequences,
        inputs=eager_input
    )
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling ConvLSTMCell.call().\n\n{e}")

print("\nTesting with Keras.Input placeholder:")
try:
    static_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        return_sequences=return_sequences,
        inputs=placeholder_input
    )
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: Exception encountered when calling ConvLSTMCell.call().\n\n{e}")