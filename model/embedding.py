import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizer, BertModel

from tqdm import tqdm
import numpy as np

from .explore import ConvBertModel

class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_size, max_len,
                 dropout=0.5, embed_type="random",
                 model_name_or_path='bert-base-chinese',
                 vector_file="sgns.wiki.bigram-char"):
        super().__init__()
        self.embed_type = embed_type
        self.embed_size = embed_size

        if embed_type == "random":
            self.embedding_layer = nn.Sequential(
                nn.Embedding(vocab_size, embed_size),
                nn.LayerNorm(normalized_shape=[max_len, embed_size]),
                nn.Dropout(dropout)
            )
        elif embed_type == "bert":
            self.bert_layer = AutoModel.from_pretrained(model_name_or_path)

        elif embed_type == "convbert":
            self.bert_layer = ConvBertModel.from_pretrained(model_name_or_path)

        elif embed_type == "token2vec":
            embedding = nn.Embedding(vocab_size, embed_size)
            embedding.from_pretrained(self.get_pretrained_vectors(vector_file))
            self.embedding_layer = nn.Sequential(
                embedding,
                nn.LayerNorm(normalized_shape=[max_len, embed_size]),
                nn.Dropout(dropout)
            )


        self.out_dim = embed_size if 'bert' not in embed_type else self.bert_layer.config.hidden_size

    def get_pretrained_vectors(self, vector_file):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        token_list = list(tokenizer.get_vocab().keys())
        vectors = [list(np.random.randn(self.embed_size)) for _ in range(tokenizer.vocab_size)]
        with open(vector_file, 'r', encoding='utf-8', errors='ignore') as reader:
            for i, line in enumerate(tqdm(reader.readlines(), desc="Read vector")):
                if i > 0:
                    line_split = line.rstrip().split(" ")
                    token = line_split.pop(0)
                    if len(token) > 1:
                        continue
                    vector = list(map(float, line_split))
                    try:
                        index = token_list.index(token)
                        vectors[index] = vector[:self.embed_size]
                    except:
                        continue
        return torch.tensor(vectors)

    def forward(self, input_ids, input_masks=None, token_type_masks=None):
        if 'bert' in self.embed_type:
            return self.bert_layer(input_ids, input_masks, token_type_masks)[0]
            # hidden_states = self.bert_layer(input_ids, input_masks, output_hidden_states=True)[2]
            # return torch.cat(hidden_states[-4:], dim=-1)
        else:
            return self.embedding_layer(input_ids)

if __name__ == '__main__':
    embedding = Embedding(vocab_size=21180, embed_size=300, max_len=512)
    embedding.get_pretrained_vectors("sgns.wiki.bigram-char")