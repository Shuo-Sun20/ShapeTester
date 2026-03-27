import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

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

# Test with eager tensor
print("Testing with eager tensor:")
eager_input = tf.random.normal((32, 10, 10, 3))
try:
    eager_output = call_func(
        inputs=eager_input,
        training=False,
        mask=None,
        axis=0,
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
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Error with eager tensor: {e}")

# Test with Keras.Input placeholder
print("\nTesting with Keras.Input placeholder:")
placeholder_input = keras.Input(shape=(10, 10, 3), batch_size=32)
try:
    static_output = call_func(
        inputs=placeholder_input,
        training=False,
        mask=None,
        axis=0,
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
        name="batch_norm_example_static",
        dtype=None
    )
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Error with placeholder: {e}")