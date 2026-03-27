import torch

def call_func(inputs, UPLO='L', out=None):
    A = inputs[0]
    return torch.linalg.eigvalsh(A, UPLO=UPLO, out=out)

A = torch.randn(3, 3, dtype=torch.complex128)
A = A + A.T.conj()
example_output = call_func([A])