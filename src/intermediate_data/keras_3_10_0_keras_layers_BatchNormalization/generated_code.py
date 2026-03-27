import keras
import numpy as np

def call_func(
    inputs,
    training=False,
    mask=None,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    synchronized=False,
    name=None,
    dtype=None
):
    layer = keras.layers.BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        synchronized=synchronized,
        name=name,
        dtype=dtype
    )
    output = layer(inputs, training=training, mask=mask)
    return output

# Create a random input tensor
input_tensor = keras.ops.convert_to_tensor(
    np.random.randn(32, 10, 10, 3).astype(np.float32)
)

# Call the function with the random input tensor
example_output = call_func(
    inputs=input_tensor,
    training=True,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    synchronized=False,
    name="batch_norm_example",
    dtype=None
)