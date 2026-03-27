import keras
import numpy as np

def call_func(
    groups=32,
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    name=None,
    dtype=None,
    inputs=None
):
    gn_layer = keras.layers.GroupNormalization(
        groups=groups,
        axis=axis,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        name=name,
        dtype=dtype
    )
    output = gn_layer(inputs)
    return output

input_tensor = keras.ops.convert_to_tensor(np.random.randn(2, 32, 32, 64).astype(np.float32))
example_output = call_func(groups=8, inputs=input_tensor)