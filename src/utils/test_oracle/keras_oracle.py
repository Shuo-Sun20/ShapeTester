import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections.abc import Callable
from keras.layers import Input
from .base import as_oracle,get_output_shape

def convert_to_symbolic_shape_keras(test_case: dict, call_func: Callable):
    def eager_to_symbolic(tensor):
        shape = tensor.shape
        symbolic_shape = shape[1:]  # 排除批量维度
        return Input(shape=symbolic_shape)
    symbolic_test_case = test_case.copy()
    if 'inputs' in test_case:
        input_data = test_case['inputs']
        if isinstance(input_data, list):
            # Preserve structure for multiple inputs
            symbolic_input = list(eager_to_symbolic(inp) for inp in input_data)
        elif isinstance(input_data, tuple):
            symbolic_input = tuple(eager_to_symbolic(inp) for inp in input_data)
        else:
            symbolic_input = eager_to_symbolic(input_data)
        symbolic_test_case['inputs'] = symbolic_input
    return call_func(**symbolic_test_case)

def eager_shape_to_symbolic_shape(eager_shape:tuple):
    if isinstance(eager_shape, tuple) and all(isinstance(e, int) for e in eager_shape):
        return (None,) + eager_shape[1:]
    elif isinstance(eager_shape, tuple) and all(isinstance(s, tuple) for s in eager_shape):
        return tuple(eager_shape_to_symbolic_shape(s) for s in eager_shape)
    else:
        return eager_shape
    
@as_oracle
def test_keras(testcase, call_func):
    eager_shape = None
    symbolic_shape = None
    try:
        output_tensor = call_func(**testcase)
        eager_shape = get_output_shape(output_tensor)
        if eager_shape is None:
            raise ValueError("Output does not have a shape attribute.")
    except Exception as e:
        eager_shape = str(e)
    try:
        symbolic_output = convert_to_symbolic_shape_keras(testcase, call_func)
        symbolic_shape = get_output_shape(symbolic_output)
        if symbolic_shape is None:
            raise ValueError("Symbolic output does not have a shape attribute.")
    except Exception as e:
        symbolic_shape = str(e)
    if not (isinstance(eager_shape, str) or eager_shape is None):
        eager_shape = eager_shape_to_symbolic_shape(eager_shape)
    if not (isinstance(symbolic_shape, str) or symbolic_shape is None):
        symbolic_shape = eager_shape_to_symbolic_shape(symbolic_shape)
    return eager_shape, symbolic_shape

if __name__ == '__main__':

    import keras
    def call_func(inputs, axis=-1, **kwargs):
        glu_layer = keras.src.ops.Glu(axis=axis, **kwargs)
        output = glu_layer(inputs)
        return output.shape
    x = keras.random.normal(shape=(2, 4, 8))
    testcase = {'inputs': x, 'axis': -1}
    print(test_keras(testcase, call_func))