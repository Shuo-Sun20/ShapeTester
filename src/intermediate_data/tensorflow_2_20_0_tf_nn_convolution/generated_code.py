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

# Construct valid input tensors
batch_size = 2
input_spatial_shape = [5, 5]
in_channels = 3
out_channels = 4
spatial_filter_shape = [3, 3]

# Create random input tensor with shape [batch_size, 5, 5, in_channels]
input_tensor = tf.random.normal(shape=[batch_size] + input_spatial_shape + [in_channels])

# Create random filters tensor with shape [3, 3, in_channels, out_channels]
filters_tensor = tf.random.normal(shape=spatial_filter_shape + [in_channels, out_channels])

# Call the function with default parameters
example_output = call_func(
    inputs=[input_tensor, filters_tensor],
    padding="SAME",
    strides=[1, 1],
    dilations=[1, 1]
)