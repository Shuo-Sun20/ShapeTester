import keras
import numpy as np

def call_func(
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding="valid",
    data_format=None,
    output_padding=None,
    dilation_rate=(1, 1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    inputs=None
):
    layer = keras.layers.Conv3DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
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

example_input = np.random.rand(4, 10, 8, 12, 128).astype('float32')
example_output = call_func(
    filters=32,
    kernel_size=2,
    strides=2,
    activation='relu',
    inputs=example_input
)