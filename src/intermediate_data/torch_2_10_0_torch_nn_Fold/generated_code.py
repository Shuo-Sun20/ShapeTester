import torch

def call_func(output_size, kernel_size, dilation=1, padding=0, stride=1, inputs=None):
    fold_layer = torch.nn.Fold(output_size, kernel_size, dilation, padding, stride)
    return fold_layer(inputs)

input_tensor = torch.randn(1, 3 * 2 * 2, 12)
example_output = call_func(output_size=(4, 5), kernel_size=(2, 2), inputs=input_tensor)