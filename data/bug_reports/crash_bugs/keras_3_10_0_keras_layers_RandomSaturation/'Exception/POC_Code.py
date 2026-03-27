import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(
    inputs,
    factor,
    value_range=(0, 255),
    data_format=None,
    seed=None
):
    layer = keras.layers.RandomSaturation(
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    return layer(inputs)

# Create test input - eager tensor with shape (4, 32, 32, 3)
eager_input = tf.random.uniform((4, 32, 32, 3), minval=0, maxval=255, dtype=tf.float32)

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(32, 32, 3))

# Test parameters that cause the defect
factor = 0.2
value_range = [0, 255]
data_format = 'channels_first'
seed = None

print("Testing with eager tensor:")
try:
    # Call with eager tensor - this should cause the defect
    eager_output = call_func(
        inputs=eager_input,
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling RandomSaturation.call().")
    print(f"{e}")

print("\nTesting with Keras Input placeholder:")
try:
    # Call with Keras Input placeholder - this should work and give static shape
    static_output = call_func(
        inputs=input_placeholder,
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: Exception - {e}")