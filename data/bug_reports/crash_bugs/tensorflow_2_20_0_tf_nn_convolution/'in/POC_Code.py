import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(inputs, padding, strides=None, dilations=None, name=None, data_format=None):
    """
    Wrapper function for tf.nn.convolution that accepts input tensors as a list.
    
    Args:
        inputs: List containing two tensors [input_tensor, filters_tensor]
        padding: A string, either "VALID" or "SAME"
        strides: Optional sequence of N ints >= 1
        dilations: Optional sequence of N ints >= 1
        name: Optional name for the operation
        data_format: Optional string specifying data format
        
    Returns:
        Tensor result of convolution
    """
    input_tensor, filters_tensor = inputs
    return tf.nn.convolution(
        input=input_tensor,
        filters=filters_tensor,
        padding=padding,
        strides=strides,
        dilations=dilations,
        name=name,
        data_format=data_format
    )

# Create test inputs that reproduce the defect
input_tensor = tf.constant([[[[1.0, 2.0, 3.0] for _ in range(5)] for _ in range(5)] for _ in range(2)])
filters_tensor = tf.constant([[[[1.0, 1.0, 1.0, 1.0] for _ in range(3)] for _ in range(3)] for _ in range(3)])

inputs = [input_tensor, filters_tensor]
padding = 'VALID'
strides = [1, 1]
dilations = [3, 3]
data_format = 'NHWC'

print("Input tensor shape:", input_tensor.shape)
print("Filters tensor shape:", filters_tensor.shape)

# Test direct function call (dynamic execution)
try:
    dynamic_result = call_func(inputs, padding, strides, dilations, data_format=data_format)
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic execution error:", e)

# Test tf.function wrapped call (static execution)
try:
    static_func = tf.function(call_func)
    static_result = static_func(inputs, padding, strides, dilations, data_format=data_format)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static execution error:", e)