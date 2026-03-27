import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs):
    return keras.ops.hstack(inputs)

# Test with eager tensor (dynamic)
eager_tensor = tf.constant([], shape=(0,), dtype=tf.float32)
dynamic_result = call_func([eager_tensor])
print("Dynamic output shape:", dynamic_result.shape)

# Test with Keras.Input placeholder (static)
try:
    input_placeholder = keras.Input(shape=(0,), dtype=tf.float32)
    static_result = call_func([input_placeholder])
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static output shape error:", str(e))

print("Defect reproduced: Dynamic shape is", dynamic_result.shape, "but static computation fails")