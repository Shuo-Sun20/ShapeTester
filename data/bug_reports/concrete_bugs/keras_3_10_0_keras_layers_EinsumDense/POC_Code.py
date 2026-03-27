import keras
import numpy as np
import tensorflow as tf
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

# Test with eager tensor
eager_input = np.random.random((5, 32, 128))
dynamic_output = call_func(
    equation='abc,cd->abd',
    output_shape=[None, 64, 32],
    activation=None,
    bias_axes='d',
    kernel_initializer=None,
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
    lora_alpha=None,
    inputs=eager_input
)

# Test with placeholder
placeholder_input = keras.Input(shape=(32, 128))
static_output = call_func(
    equation='abc,cd->abd',
    output_shape=[None, 64, 32],
    activation=None,
    bias_axes='d',
    kernel_initializer=None,
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
    lora_alpha=None,
    inputs=placeholder_input
)

print("Dynamic output shape (eager tensor):", dynamic_output.shape)
print("Static output shape (placeholder):", static_output.shape)
print("Shapes are consistent:", dynamic_output.shape == static_output.shape)