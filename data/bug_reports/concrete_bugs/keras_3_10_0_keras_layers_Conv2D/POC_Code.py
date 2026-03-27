import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras

def call_func(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
):
    layer = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint
    )
    return layer(inputs)

# Test with eager tensor
eager_input = np.random.rand(4, 10, 10, 128)
dynamic_output = call_func(
    inputs=eager_input,
    filters=1,
    kernel_size=[11, 11],
    strides=[2, 2],
    padding='valid',
    data_format=None,
    dilation_rate=1,
    groups=1,
    activation='relu',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)

# Test with Keras.Input placeholder
placeholder_input = keras.Input(shape=(10, 10, 128))
static_output = call_func(
    inputs=placeholder_input,
    filters=1,
    kernel_size=[11, 11],
    strides=[2, 2],
    padding='valid',
    data_format=None,
    dilation_rate=1,
    groups=1,
    activation='relu',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)

print(f"Dynamic output shape (eager tensor): {dynamic_output.shape}")
print(f"Static output shape (Keras.Input): {static_output.shape}")
print(f"Dynamic output shape normalized: [None, {dynamic_output.shape[1]}, {dynamic_output.shape[2]}, {dynamic_output.shape[3]}]")
print(f"Static output shape normalized: [None, {static_output.shape[1]}, {static_output.shape[2]}, {static_output.shape[3]}]")