import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(factor, data_format, seed, inputs):
    layer = keras.layers.RandomGrayscale(factor=factor, data_format=data_format, seed=seed)
    output = layer(inputs)
    return output

# Test input that causes the defect
factor = 0.5
data_format = 'channels_first'
seed = 42

# Create test data with shape (4, 32, 32, 3) as eager tensor
inputs_eager = tf.constant(np.random.random((4, 32, 32, 3)).astype(np.float32))

# Create Keras Input placeholder with same shape
inputs_placeholder = keras.Input(shape=(32, 32, 3))

print("Testing with eager tensor:")
try:
    output_eager = call_func(factor, data_format, seed, inputs_eager)
    print(f"Dynamic output shape: {output_eager.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling RandomGrayscale.call().\n{e}")

print("\nTesting with Keras Input placeholder:")
try:
    output_static = call_func(factor, data_format, seed, inputs_placeholder)
    print(f"Static output shape: {output_static.shape}")
except Exception as e:
    print(f"Static output shape: Exception encountered: {e}")