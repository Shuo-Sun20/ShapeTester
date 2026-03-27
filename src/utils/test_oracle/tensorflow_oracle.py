import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections.abc import Callable
from tensorflow import function as tf_function
from .base import as_oracle,get_output_shape

def convert_to_symbolic_shape_tf(test_case: dict, call_func: Callable):
    return tf_function(call_func)(**test_case)
    

@as_oracle
def test_tf(testcase, call_func):
    eager_shape = None
    symbolic_shape = None
    try:
        eager_output = call_func(**testcase)
        eager_shape = get_output_shape(eager_output)
        if eager_shape is None:
            raise ValueError("Output does not have a shape attribute.")
    except Exception as e:
        eager_shape = str(e)
    try:
        symbolic_output = convert_to_symbolic_shape_tf(testcase, call_func)
        symbolic_shape = get_output_shape(symbolic_output)
        if symbolic_shape is None:
            raise ValueError("Symbolic output does not have a shape attribute.")
    except Exception as e:
        symbolic_shape = str(e)
    print(eager_shape, symbolic_shape)
    return eager_shape, symbolic_shape

if __name__ == '__main__':
    import tensorflow as tf
    def call_func(input, weight, bias=None, **kwargs):
        output = tf.nn.conv1d(input, weight, stride=1, padding='VALID')
        return output.shape
    x = tf.random.normal((2, 4, 4))
    w = tf.random.normal((3, 4, 4))
    b = tf.random.normal((3,))
    testcase = {'input': x, 'weight': w, 'bias': b}
    print(test_tf(testcase, call_func))