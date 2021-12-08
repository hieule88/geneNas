
# # # import torch
# # # import torch.nn as nn
# # # import numpy as np

# # # loss_fct = nn.CrossEntropyLoss()

# # # labels =[1,0,3,2]
# # # logits =[[0.1,0.2,0.1,0.6],[0.2,0.4,0.1,0.1],[0.2,0.4,0.1,0.1],[0.2,0.4,0.1,0.1]]
# # # labels = torch.Tensor(labels)
# # # logits = torch.Tensor(logits)

# # # labels = nn.functional.one_hot(labels.to(torch.int64),4).to(torch.float32)

# # # print('labels: ',labels)
# # # print('logits: ',logits)

# # # loss = loss_fct(logits, labels)
# # # print(loss)
# # from numpy import array
# # from tensorflow.keras.utils import to_categorical

# # # define example
# # data = [[1, 3, 2, 0, 3, 2, 2, 1, 0, 1],[1, 3, 2, 0, 3, 2, 2, 1, 0, 1]]
# # data = array(data)
# # print(data)
# # # one hot encode
# # encoded = to_categorical(data, num_classes=5)
# # print(encoded)

# import torch 
# import datasets

# labels = [3,0]
# preds = [1,0]

# labels = torch.Tensor(labels)
# preds = torch.Tensor(preds)

# print(labels.shape)
# metric = datasets.load_metric("accuracy")

# result = metric.compute(predictions=preds, references=labels)

# print(result)

# import torch.nn as nn
# import torch
# from tensorflow.python.keras.utils.np_utils import to_categorical
# # Example of target with class indices
# loss = nn.CrossEntropyLoss()

# input = [[[0.1,0.2,0.7],[0.2,0.3,0.5]],[[0.5,0.1,0.4],[0.3,0.4,0.3]]]
# input_tensor = torch.Tensor(input)

# target = [[1,1],[2,0]]
# target_onehot = to_categorical(target, 3)

# target_tensor = torch.Tensor(target_onehot)

# output = loss(input_tensor, target_tensor)
# print(output)

# avg_loss = 0
# for i in range(len(input)):
#     print(input[i])
#     each_input = torch.Tensor(input[i])
#     each_target = torch.Tensor(target_onehot[i])
#     avg_loss = loss(each_input, each_target)
#     print(avg_loss)
# print(avg_loss/2)

a = [5,2,3]
for i,b in enumerate(a):
    print(i)
    print(b)
    break