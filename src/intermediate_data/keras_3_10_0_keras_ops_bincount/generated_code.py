import keras

def call_func(inputs, minlength=0, sparse=False):
    if isinstance(inputs, list):
        if len(inputs) == 2:
            x = inputs[0]
            weights = inputs[1]
        else:
            x = inputs[0]
            weights = None
    else:
        x = inputs
        weights = None
    
    return keras.ops.bincount(
        x=x,
        weights=weights,
        minlength=minlength,
        sparse=sparse
    )

x = keras.random.randint(shape=(15,), minval=0, maxval=10)
weights = keras.random.uniform(shape=(15,))
example_output = call_func([x, weights], minlength=12)