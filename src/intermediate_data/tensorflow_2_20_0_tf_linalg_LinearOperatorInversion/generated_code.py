import tensorflow as tf

def call_func(operator, inputs, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name=None):
    inv_op = tf.linalg.LinearOperatorInversion(
        operator=operator,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return inv_op.matmul(inputs[0] if isinstance(inputs, list) else inputs)

operator_matrix = tf.random.normal(shape=[2, 2])
operator = tf.linalg.LinearOperatorFullMatrix(operator_matrix)
input_tensor = tf.random.normal(shape=[2, 4])
example_output = call_func(operator, [input_tensor])