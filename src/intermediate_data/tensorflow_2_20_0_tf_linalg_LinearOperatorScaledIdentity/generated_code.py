import tensorflow as tf

def call_func(num_rows, multiplier, is_non_singular=None, is_self_adjoint=None, 
              is_positive_definite=None, is_square=True, name=None, inputs=None):
    operator = tf.linalg.LinearOperatorScaledIdentity(
        num_rows=num_rows,
        multiplier=multiplier,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return operator.matmul(inputs[0])

multiplier = tf.constant([2.0, 3.0], dtype=tf.float32)
input_tensor = tf.random.normal(shape=[2, 5, 4])
example_output = call_func(num_rows=5, multiplier=multiplier, inputs=[input_tensor])