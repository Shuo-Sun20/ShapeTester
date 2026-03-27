import tensorflow as tf

def call_func(inputs, tol=None, validate_args=False, name='matrix_rank'):
    a = inputs[0] if isinstance(inputs, list) else inputs
    return tf.linalg.matrix_rank(a=a, tol=tol, validate_args=validate_args, name=name)

random_tensor = tf.random.uniform(shape=(3, 3), minval=-1.0, maxval=1.0)
example_output = call_func(inputs=[random_tensor])