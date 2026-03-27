import keras
import numpy as np

def call_func(
    inputs,
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None
):
    layer = keras.layers.LayerNormalization(
        axis=axis,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint
    )
    return layer(inputs)

# Create random input tensor
example_input = np.random.randn(2, 5, 10, 8).astype(np.float32)
example_output = call_func(example_input, axis=[2, 3])