import tensorflow as tf

def call_func(inputs, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=True, name=None):
    diag = inputs[0]
    operator = tf.linalg.LinearOperatorDiag(
        diag=diag,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    matmul_input = inputs[1]
    return operator.matmul(matmul_input)

diag_tensor = tf.random.normal(shape=[3, 4])
matmul_tensor = tf.random.normal(shape=[3, 4, 2])
example_output = call_func([diag_tensor, matmul_tensor])