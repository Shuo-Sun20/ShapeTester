import tensorflow as tf

def call_func(inputs, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name='LinearOperatorFullMatrix'):
    matrix = inputs
    operator = tf.linalg.LinearOperatorFullMatrix(
        matrix=matrix,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return operator.to_dense()

matrix = tf.random.normal(shape=[2, 3, 5, 5])
example_output = call_func(matrix, is_non_singular=True, is_self_adjoint=False)