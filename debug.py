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

a = ['a','b','c']
dicf = {i : 0 for i in a}
print(dicf)