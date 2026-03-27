import tensorflow as tf
import numpy as np

def call_func(inputs, k, sorted=True, index_type=tf.int32, name=None):
    return tf.math.top_k(input=inputs, k=k, sorted=sorted, index_type=index_type, name=name)

# Generate random input tensor
input_tensor = tf.constant(np.random.randn(3, 4, 5), dtype=tf.float32)
k_value = 2

# Call the function and save output
result = call_func(inputs=input_tensor, k=k_value)
example_output = [result.values, result.indices]