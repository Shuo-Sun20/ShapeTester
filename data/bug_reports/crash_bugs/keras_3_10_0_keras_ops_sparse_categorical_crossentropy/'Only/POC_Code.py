import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, from_logits=False, axis=-1):
    target, output = inputs
    return keras.ops.sparse_categorical_crossentropy(target, output, from_logits=from_logits, axis=axis)

# Create test inputs as eager tensors
target_eager = tf.constant([0, 1, 2], dtype=tf.int32)  # shape=(3,)
output_eager = tf.constant([[0.9, 0.05, 0.05, 0.0],
                           [0.1, 0.8, 0.1, 0.0], 
                           [0.2, 0.3, 0.5, 0.0]], dtype=tf.float32)  # shape=(3, 4)

# Test with eager tensors (dynamic execution)
print("Testing with eager tensors:")
try:
    result_eager = call_func([target_eager, output_eager], from_logits=False, axis=-5)
    print(f"Dynamic output shape: {result_eager.shape}")
except Exception as e:
    print(f"Dynamic output shape: {str(e)}")

# Test with Keras.Input placeholders (static execution)
print("\nTesting with Keras.Input placeholders:")
target_input = keras.Input(shape=(3,), dtype='int32')
output_input = keras.Input(shape=(3, 4), dtype='float32')

try:
    result_static = call_func([target_input, output_input], from_logits=False, axis=-5)
    print(f"Static output shape: {result_static.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")