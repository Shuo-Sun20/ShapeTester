import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, name=None, dtype=None, trainable=True, autocast=True, activity_regularizer=None):
    layer = keras.layers.Identity(
        name=name, 
        dtype=dtype, 
        trainable=trainable, 
        autocast=autocast,
        activity_regularizer=activity_regularizer
    )
    return layer(inputs)

# Create test inputs
eager_tensor = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=tf.int32)
placeholder_input = keras.Input(shape=(5,), dtype='int32')

# Create activity regularizer
activity_regularizer = keras.regularizers.L2(0.01)

print("Eager tensor input shape:", eager_tensor.shape)
print("Placeholder input shape:", placeholder_input.shape)

# Test with eager tensor
try:
    dynamic_output = call_func(
        inputs=eager_tensor,
        name='test_identity',
        dtype='int32',
        trainable=True,
        autocast=True,
        activity_regularizer=activity_regularizer
    )
    print("Dynamic output shape:", dynamic_output.shape)
except Exception as e:
    print("Dynamic output error:", str(e))

# Test with placeholder
try:
    static_output = call_func(
        inputs=placeholder_input,
        name='test_identity_static',
        dtype='int32',
        trainable=True,
        autocast=True,
        activity_regularizer=activity_regularizer
    )
    print("Static output shape:", static_output.shape)
except Exception as e:
    print("Static output error:", str(e))