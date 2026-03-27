import tensorflow as tf

def call_func(
    inputs,
    is_non_singular=None,
    is_self_adjoint=None,
    is_positive_definite=None,
    is_square=None,
    name=None
):
    col, row = inputs[0], inputs[1]
    operator = tf.linalg.LinearOperatorToeplitz(
        col=col,
        row=row,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    x = inputs[2]
    return operator.matmul(x)

N = 3
col = tf.random.normal(shape=[N])
row = tf.random.normal(shape=[N])
x = tf.random.normal(shape=[N, 2])
example_output = call_func([col, row, x])