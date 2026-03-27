import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, offset=0, axis1=0, axis2=1):
    return keras.ops.trace(x=inputs, offset=offset, axis1=axis1, axis2=axis2)

# Test with eager tensor
eager_tensor = tf.constant([[1, 2, 3, 4], 
                           [5, 6, 7, 8], 
                           [9, 10, 11, 12], 
                           [13, 14, 15, 16]], dtype=tf.float32)

print("Eager tensor shape:", eager_tensor.shape)

# Test with Keras.Input placeholder
input_placeholder = keras.Input(shape=(4, 4))
print("Input placeholder shape:", input_placeholder.shape)

# Call with eager tensor (this should cause the error)
try:
    dynamic_result = call_func(eager_tensor, offset=0, axis1=-2, axis2=-2)
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic output error:", str(e))

# Call with Keras.Input placeholder
try:
    static_result = call_func(input_placeholder, offset=0, axis1=-2, axis2=-2)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static output error:", str(e))