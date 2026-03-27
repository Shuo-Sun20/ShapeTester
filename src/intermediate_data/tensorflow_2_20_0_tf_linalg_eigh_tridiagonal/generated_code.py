import tensorflow as tf

def call_func(inputs, eigvals_only=True, select='a', select_range=None, tol=None, name=None):
    alpha = inputs[0]
    beta = inputs[1]
    result = tf.linalg.eigh_tridiagonal(
        alpha=alpha,
        beta=beta,
        eigvals_only=eigvals_only,
        select=select,
        select_range=select_range,
        tol=tol,
        name=name
    )
    return result

alpha = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
beta = tf.constant([0.5, 1.0], dtype=tf.float32)
example_output = call_func([alpha, beta], eigvals_only=True)