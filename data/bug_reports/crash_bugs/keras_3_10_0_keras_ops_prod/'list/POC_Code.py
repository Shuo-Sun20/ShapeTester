import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=None, keepdims=False, dtype=None):
    return keras.ops.prod(x=inputs, axis=axis, keepdims=keepdims, dtype=dtype)

# Test with eager tensor
eager_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0], 
                           [5.0, 6.0, 7.0, 8.0], 
                           [9.0, 10.0, 11.0, 12.0]], dtype=tf.float32)

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(3, 4), dtype=tf.float32)

# Call function with eager tensor
dynamic_result = call_func(inputs=eager_tensor, axis=[-1, -2], keepdims=False, dtype='float32')
print(f"Dynamic output shape: {dynamic_result.shape}")

# Call function with Keras Input placeholder
try:
    static_result = call_func(inputs=input_placeholder, axis=[-1, -2], keepdims=False, dtype='float32')
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")

print(f"Defect reproduced: Dynamic shape {dynamic_result.shape} vs Static computation error")