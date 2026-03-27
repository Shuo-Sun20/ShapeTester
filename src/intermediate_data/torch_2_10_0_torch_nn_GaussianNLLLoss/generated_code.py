import torch
import torch.nn as nn

def call_func(full=False, eps=1e-6, reduction='mean', inputs=None):
    loss_fn = nn.GaussianNLLLoss(full=full, eps=eps, reduction=reduction)
    input_tensor, target_tensor, var_tensor = inputs
    output = loss_fn(input_tensor, target_tensor, var_tensor)
    return output

torch.manual_seed(0)
input_tensor = torch.randn(3, 2, requires_grad=True)
target_tensor = torch.randn(3, 2)
var_tensor = torch.ones(3, 2, requires_grad=True) * 0.5
inputs = [input_tensor, target_tensor, var_tensor]
example_output = call_func(inputs=inputs)