import tensorflow as tf

def call_func(inputs, data_format=None, name=None):
    """
    Calls tf.nn.bias_add with the given parameters.
    
    Args:
        inputs: A list containing two tensors [value, bias].
        data_format: A string. 'N...C' and 'NC...' are supported.
        name: A name for the operation (optional).
        
    Returns:
        A Tensor with the same type as value.
    """
    value, bias = inputs
    return tf.nn.bias_add(value, bias, data_format=data_format, name=name)

# Generate random tensors for example input
value = tf.random.normal(shape=(2, 3, 4, 5))  # 4D tensor with channel last
bias = tf.random.normal(shape=(5,))  # 1D tensor matching channel dimension

# Call function with example input
example_output = call_func(inputs=[value, bias])