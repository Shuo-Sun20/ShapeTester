import tensorflow as tf

def call_func(inputs, dtype=None, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name=None, adjoint=False, adjoint_arg=False):
    perm, x = inputs[0], inputs[1]
    if dtype is None:
        dtype = x.dtype
    operator = tf.linalg.LinearOperatorPermutation(
        perm=perm,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return operator.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

tf.random.set_seed(42)
perm = tf.constant([2, 1, 3, 0, 4])
x = tf.random.normal(shape=(5, 3))
example_output = call_func([perm, x])