import keras
import numpy as np

def call_func(
    equation,
    output_shape,
    activation=None,
    bias_axes=None,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
    lora_alpha=None,
    inputs=None
):
    layer = keras.layers.EinsumDense(
        equation=equation,
        output_shape=output_shape,
        activation=activation,
        bias_axes=bias_axes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha
    )
    output = layer(inputs)
    return output

example_input = np.random.randn(5, 32, 128).astype(np.float32)
example_output = call_func(
    equation="...x,xy->...y",
    output_shape=64,
    bias_axes="y",
    inputs=example_input
)