import tensorflow as tf

def call_func(inputs, l2_regularizer=0.0, fast=True, name=None):
    matrix, rhs = inputs[0], inputs[1]
    output = tf.linalg.lstsq(matrix, rhs, l2_regularizer=l2_regularizer, fast=fast, name=name)
    return output

# Generate random tensors for matrix and rhs
matrix = tf.random.normal(shape=[2, 5, 3])
rhs = tf.random.normal(shape=[2, 5, 2])
inputs = [matrix, rhs]

# Call the function
example_output = call_func(inputs, l2_regularizer=0.0, fast=True, name=None)