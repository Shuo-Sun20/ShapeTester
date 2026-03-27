import tensorflow as tf

def call_func(inputs, full_matrices=False, compute_uv=True, name=None):
    result = tf.linalg.svd(tensor=inputs, full_matrices=full_matrices, compute_uv=compute_uv, name=name)
    if compute_uv:
        return list(result)
    else:
        return result

example_output = call_func(inputs=tf.random.normal(shape=(5, 3)), compute_uv=True)