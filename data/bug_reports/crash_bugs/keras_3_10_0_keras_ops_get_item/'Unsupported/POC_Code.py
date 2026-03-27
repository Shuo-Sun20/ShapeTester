import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, key):
    return keras.ops.get_item(inputs, key)

# Test with eager tensor
eager_tensor = tf.constant([[[1, 2], [3, 4], [5, 6]],
                           [[7, 8], [9, 10], [11, 12]],
                           [[13, 14], [15, 16], [17, 18]],
                           [[19, 20], [21, 22], [23, 24]]])  # shape=(4, 3, 2)

key = slice(None, None, None)

print("Testing with eager tensor:")
try:
    result_eager = call_func(eager_tensor, key)
    print(f"Dynamic output shape: {result_eager.shape}")
except Exception as e:
    print(f"Error with eager tensor: {e}")

# Test with Keras.Input placeholder
print("\nTesting with Keras.Input placeholder:")
try:
    input_placeholder = keras.Input(shape=(3, 2))  # shape=(None, 3, 2) with batch dimension
    result_static = call_func(input_placeholder, key)
    print(f"Static output shape: {result_static.shape}")
except Exception as e:
    print(f"Error with placeholder: {e}")