import torch

def call_func(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, 
              norm_type=2, scale_grad_by_freq=False, sparse=False, inputs=None):
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim, 
                                   padding_idx=padding_idx, max_norm=max_norm,
                                   norm_type=norm_type, 
                                   scale_grad_by_freq=scale_grad_by_freq,
                                   sparse=sparse)
    return embedding(inputs)

# Generate random indices (must be within [0, num_embeddings-1])
num_embeddings = 10
embedding_dim = 16
input_tensor = torch.randint(0, num_embeddings, (3, 5))  # shape: (batch_size, seq_len)

example_output = call_func(num_embeddings, embedding_dim, inputs=input_tensor)