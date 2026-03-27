import torch

def call_func(input_size, hidden_size, num_layers=1, bias=True, 
              batch_first=False, dropout=0, bidirectional=False, inputs=None):
    gru = torch.nn.quantized.dynamic.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    if len(inputs) == 1:
        output = gru(inputs[0])
    else:
        output = gru(inputs[0], inputs[1])
    
    return output[0]

batch_size = 3
seq_len = 5
input_size = 10
hidden_size = 20
num_layers = 2

input_tensor = torch.randn(seq_len, batch_size, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)

example_output = call_func(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    inputs=[input_tensor, h0]
)