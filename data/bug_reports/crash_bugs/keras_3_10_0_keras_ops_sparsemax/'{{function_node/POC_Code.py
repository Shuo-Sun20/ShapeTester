import keras
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
import keras

def call_func(inputs, axis=-1):
    return keras.ops.sparsemax(inputs, axis=axis)

# Test input that causes the defect
test_input_shape = (2, 3)
axis = -4

# Create eager tensor
eager_tensor = tf.constant(np.random.randn(*test_input_shape), dtype=tf.float32)

# Create Keras Input placeholder with same shape
keras_input = keras.Input(shape=test_input_shape[1:])  # Remove batch dimension for Input

print("Testing with eager tensor:")
try:
    dynamic_output = call_func(eager_tensor, axis=axis)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

print("\nTesting with Keras Input placeholder:")
try:
    static_output = call_func(keras_input, axis=axis)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output error: {e}")