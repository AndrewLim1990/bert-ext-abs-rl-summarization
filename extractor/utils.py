from bert.utils import BERT_OUTPUT_SIZE
from collections import defaultdict
from torch import nn
from utils import compute_rouge_l

import torch

BI_LSTM_OUTPUT_SIZE = 8


class ExtractorModel(nn.Module):
    def __init__(self, bert_tokenizer, bert_model):
        super(ExtractorModel, self).__init__()
        self.input_dim = BERT_OUTPUT_SIZE
        self.bi_lstm = nn.LSTM(
            input_size=BERT_OUTPUT_SIZE,
            hidden_size=BI_LSTM_OUTPUT_SIZE,
            num_layers=1,
            bidirectional=True
        )

        self.ptr_lstm = nn.LSTM(
            input_size=BI_LSTM_OUTPUT_SIZE * 2,
            hidden_size=BI_LSTM_OUTPUT_SIZE * 2,
            num_layers=1,
            bidirectional=False
        )

        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

        self.freeze_weights(self.bert_model)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # Todo: Find suitable attn_dim
        attn_dim = self.ptr_lstm.hidden_size
        self.linear_h = torch.nn.Linear(self.ptr_lstm.hidden_size, attn_dim, bias=False)
        self.linear_e = torch.nn.Linear(self.ptr_lstm.hidden_size, attn_dim, bias=False)
        self.linear_v = torch.nn.Linear(attn_dim, 1, bias=False)
        self.tanh = torch.nn.Tanh()

        # Initial sentence embedding:
        self.init_sent_embedding = torch.nn.Parameter(torch.rand(1, attn_dim), requires_grad=True)

    @staticmethod
    def freeze_weights(model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input_embeddings, extraction_labels=None):
        """
        Todo: Convert from "dot" attention mechanism to "additive" to match paper
        Todo: Reference for the above Todo: http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture08-nmt.pdf
        Todo: Provide better comments than labeling Eq numbers from paper
        Todo: Might be better to return intermediate hidden states in order to calculate values for RL

        :param input_embeddings: shape (n_documents, seq_len, embedding_dim)
        :param extraction_labels: torch.tensor indicating whether or not the sentence should be extracted (oracle)
        :return: torch.tensor containing probability of extration of each sentence
        """
        # Add a random "[start_sentence]" embedding to the start of input_embeddings
        h, __ = self.bi_lstm(input_embeddings)
        z, __ = self.ptr_lstm(h)

        # Eq (3)
        attn_values = torch.bmm(h, z.transpose(1, 2))  # Dot attention mechanism

        # Eq (4)
        self_attn_weights = self.softmax(attn_values)

        # Eq (5)
        context = torch.bmm(self_attn_weights, h)
        batch_indicies, sent_indicies = torch.where(extraction_labels)
        label_indicies_dict = defaultdict(list)
        for row, pos in zip(batch_indicies, sent_indicies):
            label_indicies_dict[row.item()].append(pos.item())

        n_batches = h.shape[0]
        p = list()
        for batch_idx in range(n_batches):

            selected_context = context[batch_idx][label_indicies_dict[batch_idx]]
            selected_context = torch.cat([self.init_sent_embedding, selected_context])

            u = self.linear_h(h[batch_idx]) + self.linear_e(selected_context).unsqueeze(1)
            u = self.tanh(u)
            u = self.linear_v(u)
            p.append(self.sigmoid(u).squeeze())

        p = torch.nn.utils.rnn.pad_sequence(p, batch_first=True)

        return p


# Adapted from: https://github.com/ChenRocks/fast_abs_rl/tree/b1b66c180a135905574fdb9e80a6da2cabf7f15c
def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    n_art_sents = len(art_sents)
    label_vec = torch.zeros(n_art_sents)

    for abst in abs_sents:
        rouges = [compute_rouge_l(output=art_sent, reference=abst, mode='r') for art_sent in art_sents]
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break

    label_vec[extracted] = 1

    return label_vec
