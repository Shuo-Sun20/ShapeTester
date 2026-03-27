import tensorflow as tf

def call_func(inputs, name=None):
    chol, rhs = inputs[0], inputs[1]
    return tf.linalg.cholesky_solve(chol, rhs, name=name)

# Generate random positive-definite matrix and its Cholesky factor
batch_shape = (10, 2, 2)
A = tf.random.normal(batch_shape, dtype=tf.float32)
A = tf.matmul(A, A, transpose_b=True) + tf.eye(2) * 1e-3  # Make positive definite
chol = tf.linalg.cholesky(A)

# Generate random RHS
rhs = tf.random.normal((10, 2, 5), dtype=tf.float32)

example_output = call_func([chol, rhs])