import keras
import numpy as np

def call_func(inputs, from_logits=False, axis=-1):
    target, output = inputs
    return keras.ops.sparse_categorical_crossentropy(target, output, from_logits=from_logits, axis=axis)

batch_size = 3
num_classes = 4
target_tensor = keras.random.randint(shape=(batch_size,), minval=0, maxval=num_classes)
output_tensor = keras.random.uniform(shape=(batch_size, num_classes))
example_output = call_func([target_tensor, output_tensor])