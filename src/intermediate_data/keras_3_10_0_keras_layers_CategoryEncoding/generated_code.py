import numpy as np
import keras

def call_func(num_tokens, output_mode, sparse, inputs):
    layer = keras.layers.CategoryEncoding(
        num_tokens=num_tokens, 
        output_mode=output_mode, 
        sparse=sparse
    )
    if isinstance(inputs, list) and len(inputs) == 2:
        return layer(inputs[0], count_weights=inputs[1])
    else:
        return layer(inputs)

# Generate random input data
np.random.seed(42)
num_tokens = 5
input_data = np.random.randint(0, num_tokens, size=(10,))
example_output = call_func(num_tokens=num_tokens, output_mode="one_hot", sparse=False, inputs=input_data)