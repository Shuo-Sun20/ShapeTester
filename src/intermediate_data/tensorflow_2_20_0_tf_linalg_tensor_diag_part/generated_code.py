import tensorflow as tf

def call_func(inputs, name=None):
    return tf.linalg.tensor_diag_part(input=inputs, name=name)

example_input = tf.constant([[[[1111, 1112], [1121, 1122]],
                              [[1211, 1212], [1221, 1222]]],
                             [[[2111, 2112], [2121, 2122]],
                              [[2211, 2212], [2221, 2222]]]])
example_output = call_func(example_input)