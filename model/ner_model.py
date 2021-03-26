import torch
import torch.nn as nn
from .crf import CRF
from .embedding import Embedding

class NER_Model(nn.Module):

    def __init__(self, vocab_size, embed_size, num_tags,
                 max_len, device,dense_layer_type="linear",
                 dropout=0.5, embed_type="random",
                 model_name_or_path='bert-base-chinese',
                 vector_file="sgns.wiki.bigram-char"):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, embed_size, max_len, dropout,
                                         embed_type, model_name_or_path, vector_file)

        if dense_layer_type == 'linear':
            self.dense_layer = nn.Sequential(
                nn.Linear(self.embedding_layer.out_dim, num_tags*2),
                nn.GELU(),
                nn.Linear(num_tags*2, num_tags)
            )
        elif dense_layer_type == 'gru':
            self.gru = nn.GRU(input_size=self.embedding_layer.out_dim, hidden_size=32,
                              num_layers=2, batch_first=True,
                              dropout=dropout, bidirectional=True)
            self.dense_layer = nn.Linear(32*2, num_tags)

        self.crf = CRF(num_tags=num_tags, device=device)

        self.dense_layer_type = dense_layer_type

    def forward(self, token_ids, input_masks, label_ids=None):

        embeds = self.embedding_layer(token_ids, input_masks)

        if self.dense_layer_type == "linear":
            hidden_states = self.dense_layer(embeds)
        elif self.dense_layer_type == "gru":
            self.gru.flatten_parameters()
            hidden_states, _ = self.gru(embeds)
            hidden_states = self.dense_layer(hidden_states)

        else:
            hidden_states = None
            raise NotImplementedError("{} not support".format(self.dense_layer_type))

        byte_tensor = torch.empty(1, dtype=torch.uint8,
                                  device=token_ids.device)
        mask = input_masks.type_as(byte_tensor)

        if label_ids is not None:
            log_likelihood = self.crf.forward(hidden_states, label_ids, mask)
            loss = -1 * log_likelihood
        else:
            loss = torch.tensor(0, device=token_ids.device)

        sequence_tags = self.crf.decode(hidden_states, mask)

        return loss, sequence_tags