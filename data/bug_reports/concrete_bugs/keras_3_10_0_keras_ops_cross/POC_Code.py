import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1, x2 = inputs
    return keras.ops.cross(x1, x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

# Create eager tensors with shape (3, 4, 3)
eager_x1 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
eager_x2 = tf.constant([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]],
                        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]],
                        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]]])

# Test with eager tensors
eager_inputs = [eager_x1, eager_x2]
eager_result = call_func(eager_inputs, axis=None)
print("Dynamic output shape (eager tensors):", eager_result.shape)

# Test with Keras Input placeholders
input_x1 = keras.Input(shape=(4, 3))
input_x2 = keras.Input(shape=(4, 3))
static_inputs = [input_x1, input_x2]
static_result = call_func(static_inputs, axis=None)
print("Static output shape (Keras Input):", static_result.shape)

print("Defect reproduced: Dynamic shape", eager_result.shape, "!= Static shape", static_result.shape)