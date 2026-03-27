import keras

def call_func(inputs):
    # Split inputs if it's a list containing multiple tensors
    if isinstance(inputs, list) and len(inputs) > 1:
        x, y = inputs
    else:
        x = inputs if not isinstance(inputs, list) else inputs[0]
    
    # Define the forward function with custom gradient
    @keras.ops.custom_gradient
    def custom_func(*args):
        # Handle both single tensor and multiple tensors
        if len(args) > 1:
            x_val, y_val = args
            # Simple operation: element-wise multiplication
            output = keras.ops.multiply(x_val, y_val)
            
            def grad(*grad_args, upstream=None):
                if upstream is None:
                    upstream = grad_args[-1] if grad_args else None
                # Gradient: d(x*y)/dx = y, d(x*y)/dy = x
                return keras.ops.multiply(upstream, y_val), keras.ops.multiply(upstream, x_val)
        else:
            x_val = args[0]
            # Simple operation: x^2
            output = keras.ops.square(x_val)
            
            def grad(*grad_args, upstream=None):
                if upstream is None:
                    upstream = grad_args[-1] if grad_args else None
                # Gradient: d(x^2)/dx = 2x
                return keras.ops.multiply(upstream, keras.ops.multiply(2.0, x_val)),
        
        return output, grad
    
    # Call the decorated function
    if isinstance(inputs, list) and len(inputs) > 1:
        return custom_func(x, y)
    else:
        return custom_func(x)

# Generate random tensors and call the function
tensor1 = keras.random.normal(shape=(3, 4))
tensor2 = keras.random.normal(shape=(3, 4))
example_output = call_func([tensor1, tensor2])