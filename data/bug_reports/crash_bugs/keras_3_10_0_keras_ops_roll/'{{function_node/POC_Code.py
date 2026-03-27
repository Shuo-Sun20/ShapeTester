import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, shift, axis=None):
    return keras.ops.roll(inputs, shift, axis)

# Create test input - eager tensor
eager_input = tf.constant(np.random.random((2, 3, 4)), dtype=tf.float32)

# Create test input - Keras Input placeholder
placeholder_input = keras.Input(shape=(3, 4))

# Test parameters
shift = 0
axis = [0, 1]

print("Testing with eager tensor:")
try:
    dynamic_output = call_func(eager_input, shift, axis)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

print("\nTesting with Keras Input placeholder:")
try:
    static_output = call_func(placeholder_input, shift, axis)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: {e}")