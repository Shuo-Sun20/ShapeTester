import keras

def call_func(inputs, true_fn, false_fn):
    pred = inputs
    return keras.ops.cond(pred, true_fn, false_fn)

# Generate random boolean scalar for condition
pred = keras.random.uniform(shape=()) > 0.5

# Define simple functions that return random tensors
def true_fn():
    return keras.random.normal(shape=(2, 3))

def false_fn():
    return keras.random.uniform(shape=(2, 3))

# Call the function and store output
example_output = call_func(pred, true_fn, false_fn)