import torch

def call_func(kernel_size, stride, inputs, padding=0):
    input_tensor, indices_tensor = inputs[0], inputs[1]
    output_size = inputs[2] if len(inputs) > 2 else None
    
    unpool_layer = torch.nn.MaxUnpool3d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    
    if output_size is not None:
        return unpool_layer(input_tensor, indices_tensor, output_size)
    else:
        return unpool_layer(input_tensor, indices_tensor)

# Construct valid input
pool = torch.nn.MaxPool3d(kernel_size=3, stride=2, return_indices=True)
input_tensor = torch.randn(20, 16, 51, 33, 15)
output, indices = pool(input_tensor)

kernel_size = 3
stride = 2
padding = 0
inputs = [output, indices]

example_output = call_func(kernel_size, stride, [output, indices], padding)