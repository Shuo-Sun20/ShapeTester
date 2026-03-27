import tensorflow as tf

def call_func(inputs, minlength=None, maxlength=None, dtype=None, name=None, axis=None, binary_output=False):
    arr = inputs[0]
    weights = inputs[1] if len(inputs) > 1 else None
    return tf.math.bincount(arr, weights=weights, minlength=minlength, maxlength=maxlength, dtype=dtype, name=name, axis=axis, binary_output=binary_output)

example_inputs = [
    tf.constant([1, 2, 3, 1, 2, 3, 4], dtype=tf.int32)
]
example_output = call_func(example_inputs)