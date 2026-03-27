import tensorflow as tf

def call_func(inputs, adjoint=False, name=None):
    matrix = inputs[0]
    rhs = inputs[1]
    return tf.linalg.solve(matrix, rhs, adjoint=adjoint, name=name)

batch_size = 2
M = 3
K = 4
matrix_tensor = tf.random.normal(shape=[batch_size, M, M], dtype=tf.float32)
rhs_tensor = tf.random.normal(shape=[batch_size, M, K], dtype=tf.float32)
example_output = call_func([matrix_tensor, rhs_tensor])