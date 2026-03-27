import keras
import numpy as np

def call_func(subscripts, inputs):
    if isinstance(subscripts, str):
        return keras.ops.einsum(subscripts, *inputs)
    else:
        args = []
        for i, inp in enumerate(inputs):
            args.append(inp)
            args.append(subscripts[i])
        if len(subscripts) > len(inputs):
            args.append(subscripts[-1])
        return keras.ops.einsum(*args)

# Test with matrix multiplication example
matrix1 = keras.random.normal(shape=(3, 4))
matrix2 = keras.random.normal(shape=(4, 5))
example_output = call_func("ij,jk->ik", [matrix1, matrix2])