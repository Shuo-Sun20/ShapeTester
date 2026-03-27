import torch

def call_func(input_size, hidden_size, bias, inputs):
    gru_cell = torch.nn.GRUCell(input_size, hidden_size, bias)
    if len(inputs) == 1:
        output = gru_cell(inputs[0])
    else:
        output = gru_cell(inputs[0], inputs[1])
    return output

input_size = 10
hidden_size = 20
bias = True
input_tensor = torch.randn(3, input_size)
hidden_tensor = torch.randn(3, hidden_size)
inputs = [input_tensor, hidden_tensor]
example_output = call_func(input_size, hidden_size, bias, inputs)