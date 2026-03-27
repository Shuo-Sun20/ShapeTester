import tensorflow as tf

def call_func(inputs, diagonals_format='compact', transpose_rhs=False, 
              conjugate_rhs=False, name=None, partial_pivoting=True, 
              perturb_singular=False):
    diagonals = inputs[0]
    rhs = inputs[1]
    return tf.linalg.tridiagonal_solve(
        diagonals=diagonals,
        rhs=rhs,
        diagonals_format=diagonals_format,
        transpose_rhs=transpose_rhs,
        conjugate_rhs=conjugate_rhs,
        name=name,
        partial_pivoting=partial_pivoting,
        perturb_singular=perturb_singular
    )

M = 5
K = 2
batch_size = 3
compact_diagonals = tf.random.normal(shape=[batch_size, 3, M])
rhs = tf.random.normal(shape=[batch_size, M, K])
inputs = [compact_diagonals, rhs]
example_output = call_func(inputs, diagonals_format='compact')