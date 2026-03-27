import numpy as np
import keras

def call_func(
    filters,
    kernel_size,
    inputs,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer="glorot_uniform",
    pointwise_initializer="glorot_uniform",
    bias_initializer="zeros",
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None
):
    layer = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        pointwise_initializer=pointwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        pointwise_regularizer=pointwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        pointwise_constraint=pointwise_constraint,
        bias_constraint=bias_constraint
    )
    output = layer(inputs)
    return output

x = np.random.rand(4, 10, 12).astype(np.float32)
example_output = call_func(
    filters=3,
    kernel_size=4,
    strides=3,
    padding="same",
    dilation_rate=1,
    activation="relu",
    inputs=x
)