import keras
import numpy as np

def call_func(
    inputs,
    kernel_size,
    strides=1,
    padding="valid",
    depth_multiplier=1,
    data_format=None,
    dilation_rate=1,
    activation=None,
    use_bias=True,
    depthwise_initializer="glorot_uniform",
    bias_initializer="zeros",
    depthwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    bias_constraint=None
):
    layer_instance = keras.layers.DepthwiseConv1D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint
    )
    return layer_instance(inputs)

input_tensor = np.random.rand(4, 10, 12).astype(np.float32)
example_output = call_func(
    inputs=input_tensor,
    kernel_size=3,
    strides=3,
    depth_multiplier=2,
    activation='relu'
)