import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, num_classes, axis=-1, dtype=None, sparse=False):
    return keras.ops.one_hot(inputs, num_classes, axis, dtype, sparse)

# Test with eager tensor
eager_tensor = tf.constant([0, 0, 0, 0])  # shape=(4,)
print("Eager tensor shape:", eager_tensor.shape)

try:
    # Call with eager tensor
    result_eager = call_func(eager_tensor, num_classes=1, axis=1, dtype=None, sparse=False)
    print("Dynamic output shape:", result_eager.shape)
except Exception as e:
    print("Dynamic output error:", str(e))

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(4,))
print("Input placeholder shape:", input_placeholder.shape)

try:
    # Call with Keras Input placeholder
    result_static = call_func(input_placeholder, num_classes=1, axis=1, dtype=None, sparse=False)
    print("Static output shape:", result_static.shape)
except Exception as e:
    print("Static output error:", str(e))