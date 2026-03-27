import tensorflow as tf

def call_func(inputs, lower=True, adjoint=False, name=None):
    matrix, rhs = inputs
    return tf.linalg.triangular_solve(matrix, rhs, lower=lower, adjoint=adjoint, name=name)

# Generate random input tensors
tf.random.set_seed(42)
matrix = tf.linalg.band_part(tf.random.normal((4, 4), dtype=tf.float32), -1, 0)
rhs = tf.random.normal((4, 2), dtype=tf.float32)

# Call the function and store the output
example_output = call_func(inputs=[matrix, rhs], lower=True, adjoint=False)