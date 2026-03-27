import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np

def call_func(inputs, strides, padding, data_format=None, dilations=None, name=None):
    input_tensor, filter_tensor = inputs[0], inputs[1]
    return tf.nn.depthwise_conv2d(
        input=input_tensor,
        filter=filter_tensor,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )

# Create test inputs that reproduce the defect
input_tensor = tf.constant(np.random.random((2, 4, 4, 3)), dtype=tf.float32)
filter_tensor = tf.constant(np.random.random((2, 2, 3, 1)), dtype=tf.float32)

inputs = [input_tensor, filter_tensor]
strides = [1, 2, 1, 1]
padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
data_format = 'NHWC'
dilations = [4, 4]
name = None

print("Testing direct function call vs tf.function...")

# Direct function call - should work and return dynamic shape
try:
    result_direct = call_func(inputs, strides, padding, data_format, dilations, name)
    print(f"Direct call successful - Dynamic output shape: {result_direct.shape}")
except Exception as e:
    print(f"Direct call failed: {e}")

# tf.function call - should fail with static shape analysis error
try:
    compiled_func = tf.function(call_func)
    result_compiled = compiled_func(inputs, strides, padding, data_format, dilations, name)
    print(f"tf.function call successful - Static output shape: {result_compiled.shape}")
except Exception as e:
    print(f"tf.function call failed: {e}")