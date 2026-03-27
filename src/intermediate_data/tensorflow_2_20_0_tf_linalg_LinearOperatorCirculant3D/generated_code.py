import tensorflow as tf

def call_func(inputs, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name='LinearOperatorCirculant3D'):
    spectrum = inputs
    operator = tf.linalg.LinearOperatorCirculant3D(
        spectrum=spectrum,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return operator.to_dense()

spectrum = tf.complex(
    tf.random.normal([2, 3, 4], dtype=tf.float32),
    tf.random.normal([2, 3, 4], dtype=tf.float32)
)
example_output = call_func(spectrum)