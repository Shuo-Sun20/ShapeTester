import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(factor, inputs, value_range=(0, 255), data_format=None, seed=None):
    layer = keras.layers.RandomColorDegeneration(
        factor=factor,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    if isinstance(inputs, list):
        output = layer(*inputs)
    else:
        output = layer(inputs)
    return output

# Test input configuration
factor = [0.2, 0.8]
value_range = [0, 255]
data_format = 'channels_first'
seed = 42

# Create eager tensor input
eager_input = tf.constant(np.random.rand(2, 224, 224, 3).astype(np.float32))

# Create Keras.Input placeholder with same shape
placeholder_input = keras.Input(shape=(224, 224, 3))

print("Testing with eager tensor:")
try:
    dynamic_output = call_func(factor, eager_input, value_range, data_format, seed)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling RandomColorDegeneration.call().\n{e}")

print("\nTesting with Keras.Input placeholder:")
try:
    static_output = call_func(factor, placeholder_input, value_range, data_format, seed)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: Exception encountered.\n{e}")