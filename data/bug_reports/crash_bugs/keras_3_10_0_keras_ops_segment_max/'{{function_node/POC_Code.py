import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, num_segments=None, sorted=False):
    data = inputs[0]
    segment_ids = inputs[1]
    return keras.ops.segment_max(data, segment_ids, num_segments, sorted)

# Test with eager tensors (dynamic)
data_eager = tf.constant([1, 2, 10, 20, 100, 200])
segment_ids_eager = tf.constant([0, 0, 1, 1, 2, 2])

print("Dynamic execution with eager tensors:")
try:
    result_dynamic = call_func([data_eager, segment_ids_eager], num_segments=1, sorted=False)
    print(f"Dynamic output shape: {result_dynamic.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

# Test with Keras.Input placeholders (static)
data_input = keras.Input(shape=(6,))
segment_ids_input = keras.Input(shape=(6,))

print("\nStatic execution with Keras.Input placeholders:")
try:
    result_static = call_func([data_input, segment_ids_input], num_segments=1, sorted=False)
    print(f"Static output shape: {result_static.shape}")
except Exception as e:
    print(f"Static output shape: {e}")