import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=0):
    return keras.ops.stack(x=inputs, axis=axis)

# Create test inputs
eager_tensors = [
    tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
    tf.constant([[7, 8, 9], [10, 11, 12]], dtype=tf.float32),
    tf.constant([[13, 14, 15], [16, 17, 18]], dtype=tf.float32)
]

# Test with eager tensors (dynamic execution)
try:
    dynamic_result = call_func(eager_tensors, axis=-4)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {str(e)}")

# Test with Keras.Input placeholders (static execution)
input_placeholders = [
    keras.Input(shape=(2, 3)),
    keras.Input(shape=(2, 3)),
    keras.Input(shape=(2, 3))
]

try:
    static_result = call_func(input_placeholders, axis=-4)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")