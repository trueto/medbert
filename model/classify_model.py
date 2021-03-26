import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import Embedding

# refer to: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
class TextCNN(nn.Module):
    def __init__(self, hidden_size, kernel_num, kernel_sizes):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num,
                                             (K, hidden_size)) for K in kernel_sizes])

    def forward(self, hidden_states):
        # (N,Ci,W,D)
        hidden_states = hidden_states.unsqueeze(1)
        # [(N, Co, W), ...]*len(Ks)
        hidden_states = [F.relu(conv(hidden_states)).squeeze(3) for conv in self.convs]

        # [(N, Co), ...]*len(Ks)
        hidden_states = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in hidden_states]

        hidden_states = torch.cat(hidden_states, 1)

        return hidden_states

# refer to: https://github.com/keishinkickback/Pytorch-RNN-text-classification/blob/master/model.py
class TextRNN(nn.Module):

    def __init__(self, input_size, num_layers, dropout,rnn_model="LSTM", use_first=True):
        super().__init__()
        if rnn_model == "LSTM":
            self.rnn = nn.LSTM(input_size, 32, num_layers=num_layers,
                               dropout=dropout, batch_first=True, bidirectional=True)
        if rnn_model == "GRU":
            self.rnn = nn.GRU(input_size, 32, num_layers=num_layers,
                              dropout=dropout, batch_first=True, bidirectional=True)

        self.bn = nn.BatchNorm1d(64)
        self.use_first = use_first

    def forward(self, hidden_states):
        rnn_output, _ = self.rnn(hidden_states, None)

        if self.use_first:
            return self.bn(rnn_output[:, 0, :])
        else:
            return self.bn(torch.mean(rnn_output, dim=1))

class CLS_Model(nn.Module):

    def __init__(self, vocab_size, embed_size, num_labels,
                dense_layer_type="linear",
                dropout=0.5, embed_type="random", max_len=128,
                model_name_or_path="bert-base-chinese", vector_file=""):
        super().__init__()
        self.embed_layer = Embedding(vocab_size=vocab_size, embed_size=embed_size,
                                     max_len=max_len, dropout=dropout,embed_type=embed_type,
                                     model_name_or_path=model_name_or_path, vector_file=vector_file)

        if dense_layer_type == "linear":
            self.classifier = nn.Linear(self.embed_layer.out_dim, num_labels)
        elif dense_layer_type == "cnn":
            self.classifier = nn.Sequential(
                TextCNN(self.embed_layer.out_dim, kernel_num=1024, kernel_sizes=(2, 3, 4)),
                nn.Linear(1024 * 3, num_labels)
            )
        elif dense_layer_type == "gru":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                TextRNN(self.embed_layer.out_dim, dropout=dropout, num_layers=2, rnn_model="GRU",
                        use_first=False),
                nn.Linear(64, num_labels)
            )

        self.dense_layer_type = dense_layer_type
        self.num_labels = num_labels

    def forward(self, input_ids, token_type_ids,
        attention_mask, label_ids=None):

        hidden_states = self.embed_layer(input_ids, attention_mask, token_type_ids)

        if self.dense_layer_type == "linear":
            hidden_states = hidden_states[:, 0, :]

        logits = self.classifier(hidden_states)

        loss = torch.tensor(0, device=input_ids.device)
        if label_ids is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))

        return loss, logits.argmax(dim=-1)

