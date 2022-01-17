import torch.nn as nn
import torch

loss_fct = nn.CrossEntropyLoss()

a = [[0.1, 0.2, 0.7] , [0.3, 0.3, 0.4] , [0.1, 0.5, 0.4]]
a = torch.tensor(a)
b = [2, 2, 1]
b = torch.tensor(b)

print(loss_fct(a,b))