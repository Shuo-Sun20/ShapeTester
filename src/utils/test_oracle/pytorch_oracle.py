import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from collections.abc import Callable
import torch
from .base import as_oracle,get_output_shape


def dynamic_compile(test_case: dict, call_func: Callable):
    new_call_func = torch.compile(call_func, dynamic=True)
    # symbolic_test_case = test_case.copy()
    # for key, value in test_case.items():
    #     if hasattr(value, 'device'):
    #         # Move tensors to meta device to avoid real allocation
    #         symbolic_test_case[key] = value.to('meta')
    return new_call_func(**test_case)

def move_to_meta(test_case):
    if hasattr(test_case, 'device'):
        return test_case.to('meta')
    elif isinstance(test_case, dict):
        return {key: move_to_meta(value) for key, value in test_case.items()}
    elif isinstance(test_case, list):
        return [move_to_meta(item) for item in test_case]
    elif isinstance(test_case, tuple):
        return tuple(move_to_meta(item) for item in test_case)
    else:
        return test_case

def meta_tensor(test_case: dict, call_func: Callable):
    symbolic_test_case = test_case.copy()
    for key, value in test_case.items():
        symbolic_test_case[key] = move_to_meta(value)

    return call_func(**symbolic_test_case)

@as_oracle
def test_pytorch(testcase, call_func):
    eager_shape = None
    symbolic_shape = None
    meta_shape = None
    try:
        eager_output = call_func(**testcase)
        eager_shape = get_output_shape(eager_output)
        if eager_shape is None:
            raise ValueError("Output does not have a shape attribute.")
    except Exception as e:
        eager_shape = str(e)
    try:
        symbolic_output = dynamic_compile(testcase, call_func)
        symbolic_shape = get_output_shape(symbolic_output)
        if symbolic_shape is None:
            raise ValueError("Symbolic output does not have a shape attribute.")
    except Exception as e:
        symbolic_shape = str(e)
    try:
        meta_output = meta_tensor(testcase, call_func)
        meta_shape = get_output_shape(meta_output)
        if meta_shape is None:
            raise ValueError("Meta output does not have a shape attribute.")
    except Exception as e:
        meta_shape = str(e)
    return eager_shape, symbolic_shape, meta_shape

if __name__ == '__main__':
    import torch.nn.functional as F
    def call_func(input, weight, bias=None, **kwargs):
        output = F.linear(input, weight, bias)
        return output.shape
    x = torch.randn(2, 4)
    w = torch.randn(3, 4)
    b = torch.randn(3)
    testcase = {'input': x, 'weight': w, 'bias': b}
    print(test_pytorch(testcase, call_func))