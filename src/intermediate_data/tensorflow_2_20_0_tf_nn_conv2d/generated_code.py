import tensorflow as tf

def call_func(inputs, strides, padding, data_format='NHWC', dilations=1, name=None):
    """
    Calls tf.nn.conv2d with given parameters.
    
    Args:
        inputs: List containing two tensors [input_tensor, filters_tensor]
                input_tensor: Shape [batch, height, width, channels] (NHWC) 
                             or [batch, channels, height, width] (NCHW)
                filters_tensor: Shape [filter_height, filter_width, in_channels, out_channels]
        strides: List of ints [batch_stride, height_stride, width_stride, channel_stride]
        padding: String 'SAME' or 'VALID' or list of explicit paddings
        data_format: 'NHWC' or 'NCHW'
        dilations: List of ints [batch_dilation, height_dilation, width_dilation, channel_dilation]
        name: Optional operation name
    
    Returns:
        Convolution output tensor
    """
    input_tensor, filters_tensor = inputs
    return tf.nn.conv2d(
        input=input_tensor,
        filters=filters_tensor,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations    )

# Create example input tensors
batch_size = 2
in_channels = 3
out_channels = 4
height = 32
width = 32
filter_height = 3
filter_width = 3

# Generate random input tensor (NHWC format)
input_tensor = tf.random.normal(shape=[batch_size, height, width, in_channels], dtype=tf.float32)

# Generate random filters tensor
filters_tensor = tf.random.normal(
    shape=[filter_height, filter_width, in_channels, out_channels], 
    dtype=tf.float32
)

# Call function with VALID padding and stride 1
example_output = call_func(
    inputs=[input_tensor, filters_tensor],
    strides=[1, 1, 1, 1],
    padding='VALID'
)