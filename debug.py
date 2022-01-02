import torch
from torchcrf import CRF
import torch.nn as nn

a = torch.tensor([[ 0,  0,  3, -2, -2, -2],\
        [ 5,  0, -2, -2, -2, -2],\
        [ 7,  8,  0, -2, -2, -2]])

b = [[1 if a[j][i] != -2 else 0 for i in range(len(a[j]))] for j in range(len(a))]

print(b)