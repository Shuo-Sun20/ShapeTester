import keras
import numpy as np

def call_func(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
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
    layer = keras.layers.SeparableConv2D(
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
    return layer(inputs)

example_input = np.random.rand(2, 32, 32, 16).astype(np.float32)
example_output = call_func(
    inputs=example_input,
    filters=32,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    activation="relu"
)