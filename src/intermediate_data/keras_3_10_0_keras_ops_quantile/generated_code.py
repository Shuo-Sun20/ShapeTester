import keras
import numpy as np

def call_func(inputs, q, axis=None, method="linear", keepdims=False):
    return keras.ops.quantile(x=inputs, q=q, axis=axis, method=method, keepdims=keepdims)

example_output = call_func(
    inputs=keras.random.uniform(shape=(5, 4, 3), minval=0.0, maxval=1.0),
    q=[0.25, 0.5, 0.75],
    axis=1,
    method="midpoint",
    keepdims=True
)