import keras

def call_func(inputs, from_logits=False):
    target, output = inputs[0], inputs[1]
    return keras.ops.binary_crossentropy(target, output, from_logits=from_logits)

# Generate random tensors with matching shapes
target_tensor = keras.ops.round(keras.random.uniform((4,), 0.0, 1.0, dtype="float32"))
output_tensor = keras.random.uniform((4,), 0.0, 1.0, dtype="float32")
inputs = [target_tensor, output_tensor]

# Call the function and save output
example_output = call_func(inputs, from_logits=False)