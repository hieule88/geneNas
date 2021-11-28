import torch.nn as nn
from collections import Counter
import numpy as np
import datasets
import torch 

dataset =  datasets.load_dataset('conll2003')

def vocab_to_ids():
    all_tokens = sum(dataset["train"]["tokens"], [])
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))

    counter = Counter(all_tokens_array)

    vocab_size = len(counter)

    vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]
    
    v_t_i = dict(zip(vocabulary, range(1, len(vocabulary)+1)))

    return v_t_i, vocabulary, vocab_size

class GloveEmbedding(nn.Module):
    def __init__(self, glove_dir, vocab):
        super().__init__()

        self.vocab = vocab
        self.glove_dir = glove_dir
        self.embed_dim = int(glove_dir.split('.')[-2][:-1]) 
        self.token_emb = self.init_token_emb()

    def init_token_emb(self):
        vocabulary = self.vocab
        embeddings_index = {} # empty dictionary of GloVe
        f = open(self.glove_dir, encoding="utf-8")

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        num_tokens = len(vocabulary) + 2
        embedding_dim = self.embed_dim

        word_index = dict(zip(vocabulary, range(1, len(vocabulary)+1))) # dict of Vocab Conll
        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))

        # Index of dict GloVe fit index of Vocab Conll + 1  
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector            

        # Calculate unk_emb
        unk_embed = np.mean(embedding_matrix,axis=0,keepdims=True) 
        embedding_matrix[-1] = unk_embed

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embedding_matrix[i] = unk_embed
                # [PAD] = embedding_matrix[0] = [0,0,...0]
                # <unk> = embedding_matrix[-1] = unk_embed

        token_emb = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix).float())
        return token_emb

    def forward(self, input_ids):
        # INPUT : torch tensor OF INDEXES OF SENTENCE
        # OUTPUT : torch tensor shape = (sentence_max_length, embed_dim) 
        
        x = self.token_emb(input_ids)
        return x 

embed = GloveEmbedding('C:\glove\glove.6B.200d.txt', vocab_to_ids()[1])

inputs = torch.Tensor([[1,2,3],[2,3,4],[3,4,5]]).to(torch.int64)
print(embed(inputs))