import keras
import numpy as np

def call_func(inputs, from_logits=False, axis=-1):
    target, output = inputs
    return keras.ops.categorical_crossentropy(target, output, from_logits=from_logits, axis=axis)

np.random.seed(0)
target_tensor = keras.ops.convert_to_tensor(np.eye(3))
output_tensor = keras.ops.convert_to_tensor(np.random.dirichlet([1, 1, 1], size=3))
example_output = call_func([target_tensor, output_tensor])