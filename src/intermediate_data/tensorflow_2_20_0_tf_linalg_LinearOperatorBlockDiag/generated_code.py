import tensorflow as tf

def call_func(
    inputs,
    is_non_singular=None,
    is_self_adjoint=None,
    is_positive_definite=None,
    is_square=None,
    name=None
):
    operators = [tf.linalg.LinearOperatorFullMatrix(tensor) for tensor in inputs]
    op = tf.linalg.LinearOperatorBlockDiag(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return op.to_dense()

tensor1 = tf.random.normal(shape=[2, 2])
tensor2 = tf.random.normal(shape=[2, 2])
example_output = call_func([tensor1, tensor2])