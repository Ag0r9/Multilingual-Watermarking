import torch

def random_hash_function(input_ids, vocab_size):
    random_hash = torch.rand(input_ids.shape[0], vocab_size)
    return random_hash

# def random_hash_function(input_ids, vocab_size):
#     embedding_size = input_ids.shape[-1]
#     random_hash = torch.randint(0, vocab_size, (embedding_size,))
#     return random_hash
