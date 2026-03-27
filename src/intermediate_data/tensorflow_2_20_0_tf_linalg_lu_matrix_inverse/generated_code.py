import tensorflow as tf

def call_func(inputs, validate_args=False, name=None):
    lower_upper = inputs[0]
    perm = inputs[1]
    return tf.linalg.lu_matrix_inverse(lower_upper=lower_upper, perm=perm, validate_args=validate_args, name=name)

x = tf.random.normal(shape=(3, 2, 2))
lu, perm = tf.linalg.lu(x)
example_output = call_func(inputs=[lu, perm])