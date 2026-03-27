import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
from keras.regularizers import L1

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
eager_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=tf.bfloat16)
placeholder_input = keras.Input(shape=(5,), dtype=tf.bfloat16)

# Test parameters
test_params = {
    'name': 'test_identity',
    'dtype': 'bfloat16',
    'trainable': False,
    'autocast': True,
    'activity_regularizer': L1(0.01)
}

print("Testing with eager tensor:")
try:
    dynamic_output = call_func(eager_tensor, **test_params)
    print(f"Dynamic output shape: {dynamic_output.shape}")
    print(f"Dynamic output dtype: {dynamic_output.dtype}")
except Exception as e:
    print(f"Dynamic output error: {e}")

print("\nTesting with placeholder:")
try:
    static_output = call_func(placeholder_input, **test_params)
    print(f"Static output shape: {static_output.shape}")
    print(f"Static output dtype: {static_output.dtype}")
except Exception as e:
    print(f"Static output error: {e}")