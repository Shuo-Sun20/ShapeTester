import torch

def call_func(blank=0, reduction='mean', zero_infinity=False, inputs=None):
    log_probs, targets, input_lengths, target_lengths = inputs
    ctc_loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    output = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    return output

# Generate example input tensors
T = 50
C = 20
N = 16
S = 30
S_min = 10

log_probs = torch.randn(T, N, C).log_softmax(2)
targets = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

inputs = [log_probs, targets, input_lengths, target_lengths]
example_output = call_func(blank=0, reduction='mean', zero_infinity=False, inputs=inputs)