import tensorflow as tf

def call_func(operators, inputs, is_non_singular=None, is_self_adjoint=None, 
              is_positive_definite=None, is_square=None, name=None):
    operator = tf.linalg.LinearOperatorKronecker(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return operator.matmul(inputs)

# Construct valid inputs
op1_matrix = tf.random.normal(shape=[2, 2])
op2_matrix = tf.random.normal(shape=[2, 2])
operator_1 = tf.linalg.LinearOperatorFullMatrix(op1_matrix)
operator_2 = tf.linalg.LinearOperatorFullMatrix(op2_matrix)
operators = [operator_1, operator_2]
inputs = tf.random.normal(shape=[4, 2])

example_output = call_func(operators, inputs)