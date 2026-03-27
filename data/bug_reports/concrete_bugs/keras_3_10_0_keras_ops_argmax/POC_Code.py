import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, axis=None, keepdims=False):
    x = inputs[0] if isinstance(inputs, list) else inputs
    return keras.ops.argmax(x=x, axis=axis, keepdims=keepdims)

# Test with eager tensor
eager_tensor = tf.random.normal((3, 4, 5))
eager_inputs = [eager_tensor]
eager_result = call_func(eager_inputs, axis=None, keepdims=True)
print("Dynamic output shape (eager):", eager_result.shape)

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(4, 5), batch_size=3)
placeholder_inputs = [input_placeholder]
placeholder_result = call_func(placeholder_inputs, axis=None, keepdims=True)
print("Static output shape (placeholder):", placeholder_result.shape)

print("Shapes match:", eager_result.shape == placeholder_result.shape)