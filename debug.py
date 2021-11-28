# import torch 
# import torch.nn as nn
# from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import TimeDistributed
# inputs = [1,2,3,4,5,6]

# out_proj = nn.Linear(3,2)

# tD = 
dataset = {
    'train': 1,
    'validation': 2,
    'test': 3
} 
abc = [x for x in dataset.keys() if "validation" in x]
print(abc)