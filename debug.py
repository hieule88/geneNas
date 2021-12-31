import torch
from torchcrf import CRF
num_tags = 5  # number of tags is 5
model = CRF(num_tags, batch_first=True)
seq_length = 3  # maximum sequence length in a batch
batch_size = 2  # number of samples in the batch
emissions = torch.randn(batch_size, seq_length, num_tags)
tags = torch.tensor([[0, 2, 3], [1, 4, 1]], dtype=torch.long)  # (seq_length, batch_size)
print(emissions.shape)
print(tags.shape)
print(model(emissions, tags))
print(model.decode(emissions))