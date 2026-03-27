import keras
import numpy as np
import tensorflow as tf
import keras
from keras import ops
import tensorflow as tf

def call_func(index, branches, inputs):
    return ops.switch(index, branches, *inputs)

# Define the lambda functions that will be used as branches
branch_0 = lambda x, y: ops.sum(x, axis=0)  # Sum along first axis, output shape: (3,)
branch_1 = lambda x, y: ops.sum(x, axis=1, keepdims=True)  # Sum along second axis with keepdims, output shape: (2, 1)

branches = [branch_0, branch_1]

# Test with eager tensors
eager_tensor_1 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape: (2, 3)
eager_tensor_2 = tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # shape: (2, 3)
eager_inputs = [eager_tensor_1, eager_tensor_2]

# Call with eager tensors
eager_result = call_func(1, branches, eager_inputs)
print("Dynamic output shape (eager):", eager_result.shape)

# Test with Keras.Input placeholders
input_1 = keras.Input(shape=(3,), batch_size=2)  # shape: (2, 3)
input_2 = keras.Input(shape=(3,), batch_size=2)  # shape: (2, 3)
placeholder_inputs = [input_1, input_2]

# Call with placeholders
placeholder_result = call_func(1, branches, placeholder_inputs)
print("Static output shape (placeholder):", placeholder_result.shape)

print("Defect reproduced: Dynamic shape {} != Static shape {}".format(
    eager_result.shape, placeholder_result.shape))