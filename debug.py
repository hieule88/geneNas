import torch 
import torch.nn as nn

# import datasets
# from transformers import AutoTokenizer
# import numpy as np
# from collections import Counter
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import TimeDistributed

input = torch.Tensor([[2, 3], [4, 6], [0,1]])
# print(input.shape)
print(input.shape)
output = nn.Linear(2,1)(input)
print(output.size())
# # tokenizer = AutoTokenizer.from_pretrained(
# #             model_name_or_path, use_fast=True
# #         )

# # def vocab_to_ids(data):
# #     all_tokens = sum(data["train"]["tokens"], [])
# #     all_tokens_array = np.array(list(map(str.lower, all_tokens)))

# #     counter = Counter(all_tokens_array)

# #     vocab_size = len(counter)

# #     vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]
    
# #     v_t_i = dict(zip(vocabulary, range(1, len(vocabulary)+1)))
# #     return v_t_i, vocab_size

# # data = datasets.load_dataset('conll2003')
# # vocab = vocab_to_ids(data)
# # v_t_ids = vocab[0]
# # len_vocab = vocab[1]

# # def convert_to_features(example_batch, indices=None):

# #     texts_or_text_pairs = example_batch['tokens']

# #     # # lower -> to ids 
# #     # for i in range(len(texts_or_text_pairs)):
# #     #     for j in range(len(texts_or_text_pairs[i])):
# #     #         id = v_t_ids.get(texts_or_text_pairs[i][j].lower())
# #     #         if id is not None:
# #     #             texts_or_text_pairs[i][j] = id
# #     #         else :
# #     #             texts_or_text_pairs[i][j] = len_vocab

# #     for i in range(len(texts_or_text_pairs)):
# #         texts_or_text_pairs[i] = ' '.join(texts_or_text_pairs[i])
    
# #     # Tokenize the text/text pairs
# #     # if self.task_name != 'ner':
# #     features = tokenizer.batch_encode_plus(
# #         texts_or_text_pairs,
# #         max_length=128,
# #         pad_to_max_length=True,
# #         truncation=True,
# #     )

# #     # else:
# #     #     features = 

# #     # Rename label to labels to make it easier to pass to model forward
# #     features["labels"] = example_batch['ner_tags']

# #     return features

# # def main():    
# #     features = convert_to_features(data['train'][0:2])
# #     print(features)


# # if __name__ == '__main__':
# #     main()


# import datasets
# from datasets import features
# from tensorflow.python.eager.context import set_global_seed
# from transformers import AutoTokenizer
# import numpy as np
# from collections import Counter

# def vocab_to_ids(data):
#     all_tokens = sum(data["train"]["tokens"], [])
#     all_tokens_array = np.array(list(map(str.lower, all_tokens)))

#     counter = Counter(all_tokens_array)

#     vocab_size = len(counter)

#     vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]
    
#     v_t_i = dict(zip(vocabulary, range(1, len(vocabulary)+1)))
#     return v_t_i, vocab_size

# data = datasets.load_dataset('conll2003')
# vocab = vocab_to_ids(data)
# v_t_ids = vocab[0]
# len_vocab = vocab[1]

# def convert_to_features(example_batch, indices=None):
#     max_length = 128
#     texts_or_text_pairs = example_batch['tokens']

#     # lower -> to ids 
#     input_ids = []
#     sentence = []
#     for i in range(len(texts_or_text_pairs)):
#         for j in range(len(texts_or_text_pairs[i])):
#             id = v_t_ids.get(texts_or_text_pairs[i][j].lower())
#             if id is not None:
#                 sentence.append(id)
#             else :
#                 sentence.append(len_vocab)
#         for j in range(len(texts_or_text_pairs[i]), max_length):
#             sentence.append(0)
#         input_ids.append(sentence)

#     features = {}
#     features['input_ids'] = input_ids


#     # Rename label to labels to make it easier to pass to model forward
#     features["labels"] = example_batch['ner_tags']

#     return features

# def main():    
#     features = convert_to_features(data['train'][0:2])
#     print(features)


# if __name__ == '__main__':
#     main()

