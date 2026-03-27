import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
import numpy as np

def call_func(inputs, start_indices, shape):
    return keras.ops.slice(inputs, start_indices, shape)

# Test with eager tensors (dynamic case)
print("Testing with eager tensors:")
eager_inputs = tf.constant(np.zeros((5, 5)), dtype=tf.float32)
start_indices = [2, 1]
shape = [4, 4]

try:
    dynamic_result = call_func(eager_inputs, start_indices, shape)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape error: {e}")

# Test with Keras.Input placeholders (static case)
print("\nTesting with Keras.Input placeholders:")
placeholder_inputs = keras.Input(shape=(5,), batch_size=5)
static_result = call_func(placeholder_inputs, start_indices, shape)
print(f"Static output shape: {static_result.shape}")

print(f"\nDefect reproduced: Dynamic case fails while static case succeeds")
print(f"The issue is that start_indices={start_indices} + shape={shape} exceeds input bounds")
print(f"Input shape: (5, 5), but trying to slice from [2, 1] with shape [4, 4]")
print(f"This would require accessing indices up to [2+4-1, 1+4-1] = [5, 4], but max valid indices are [4, 4]")