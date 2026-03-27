import tensorflow as tf

def call_func(inputs, name=None, k=0, num_rows=None, num_cols=None, padding_value=0, align="RIGHT_LEFT"):
    return tf.linalg.diag(diagonal=inputs, name=name, k=k, num_rows=num_rows, num_cols=num_cols, padding_value=padding_value, align=align)

example_input = tf.constant([[1, 2, 3], [4, 5, 6]])
example_output = call_func(inputs=example_input, k=1, num_rows=4, num_cols=4)