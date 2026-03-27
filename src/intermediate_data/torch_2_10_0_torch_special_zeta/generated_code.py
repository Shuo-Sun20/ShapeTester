import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0] if isinstance(inputs[0], torch.Tensor) else torch.tensor(inputs[0])
    other_tensor = inputs[1] if isinstance(inputs[1], torch.Tensor) else torch.tensor(inputs[1])
    return torch.special.zeta(input=input_tensor, other=other_tensor, out=out)

torch.manual_seed(42)
example_inputs = [torch.rand(2, 3) * 2 + 1, torch.rand(2, 3) + 0.5]
example_output = call_func(example_inputs)