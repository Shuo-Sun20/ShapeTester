import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, indexing="xy"):
    return keras.ops.meshgrid(*inputs, indexing=indexing)

# Create eager tensors with shapes (3,), (4,), (5,)
eager_tensor1 = tf.constant([1, 2, 3])  # shape (3,)
eager_tensor2 = tf.constant([4, 5, 6, 7])  # shape (4,)
eager_tensor3 = tf.constant([8, 9, 10, 11, 12])  # shape (5,)

# Test with eager tensors
print("Testing with eager tensors:")
eager_inputs = [eager_tensor1, eager_tensor2, eager_tensor3]
eager_result = call_func(eager_inputs, indexing="xy")
print("Dynamic output shapes:")
for i, tensor in enumerate(eager_result):
    print(f"Tensor {i}: {tensor.shape}")

# Test with Keras.Input placeholders of the same shapes
print("\nTesting with Keras.Input placeholders:")
input1 = keras.Input(shape=(3,))
input2 = keras.Input(shape=(4,))
input3 = keras.Input(shape=(5,))

static_inputs = [input1, input2, input3]
static_result = call_func(static_inputs, indexing="xy")
print("Static output shapes:")
for i, tensor in enumerate(static_result):
    print(f"Tensor {i}: {tensor.shape}")

print("\nDefect reproduced: Dynamic and static output shapes are inconsistent!")