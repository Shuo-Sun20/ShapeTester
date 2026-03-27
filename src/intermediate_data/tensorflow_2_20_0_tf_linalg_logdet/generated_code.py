import tensorflow as tf

def call_func(inputs, name=None):
    matrix = inputs[0]
    return tf.linalg.logdet(matrix=matrix, name=name)

# Generate random positive definite matrix for input
A = tf.random.normal(shape=(5, 5))
matrix = tf.matmul(A, A, transpose_b=True) + tf.eye(5) * 0.1  # Make positive definite
example_output = call_func(inputs=[matrix])