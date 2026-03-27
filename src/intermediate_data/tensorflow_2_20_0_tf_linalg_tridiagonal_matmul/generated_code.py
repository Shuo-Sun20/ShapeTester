import tensorflow as tf

def call_func(inputs, diagonals_format='compact', name=None):
    diagonals = inputs[0]
    rhs = inputs[1]
    return tf.linalg.tridiagonal_matmul(diagonals, rhs, diagonals_format=diagonals_format, name=name)

# Generate random tensors for the 'sequence' format
superdiag = tf.random.uniform(shape=[5], dtype=tf.float32)
maindiag = tf.random.uniform(shape=[5], dtype=tf.float32)
subdiag = tf.random.uniform(shape=[5], dtype=tf.float32)
diagonals = [superdiag, maindiag, subdiag]
rhs = tf.random.uniform(shape=[5, 3], dtype=tf.float32)

example_output = call_func(inputs=[diagonals, rhs], diagonals_format='sequence')