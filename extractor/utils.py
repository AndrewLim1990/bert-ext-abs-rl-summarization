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
        rand_floats = torch.FloatTensor(1, attn_dim).uniform_(-1, 1) * 1e-1
        self.init_sent_embedding = torch.nn.Parameter(rand_floats, requires_grad=True)
        self.padding_value = -69

    @staticmethod
    def freeze_weights(model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, sent_embeddings, sent_mask, extraction_indicator=None, use_init_embedding=True):
        """
        Obtains binary probability of extraction for each sentence within the input sent_embeddings

        Note:
         - Equation number comments reference http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture08-nmt.pdf
         - Using dot attention mechanism instead of additive. You can change this if you want (Eq3)

        :param sent_embeddings:      Shape (batch_size, n_src_sents, embedding_dim)
        :param sent_mask:            A torch.tensor (bool) indicating if the embedding actually exists in the source
                                     doc. Necessary because operating in batches. Shape: (batch_size, n_src_sents)
        :param extraction_indicator: A torch.tensor (bool) indicating which embeddings to serve as an input in order to
                                     determine extraction probs amongst sents for the NEXT sent to extract. If None,
                                     use_init_embedding MUST be True in order to have an intial embedding.
                                     Shape: (n_documents, n_src_sents)
        :param use_init_embedding:   A bool indicating whether or not to use initial embedding
        :return:
          - p:          torch.tensor of shape (batch_size, n_ext_sents, n_src_sents) containing probability of
                        extracting each sentence individually (not a distribution, just binary)
          - p_mask:     torch.tensor of shape (batch_size, n_ext_sents) indicating if entries in 'p' are should be
                        considered legitmate. Necessary because operating in batches
        """
        h, __ = self.bi_lstm(sent_embeddings)
        z, __ = self.ptr_lstm(h)

        # Eq (3) Self attention: Dot attention mechanism
        attn_values = torch.bmm(h, z.transpose(1, 2))

        # Eq (4)
        self_attn_weights = self.softmax(attn_values)

        # Eq (5)
        context = torch.bmm(self_attn_weights, h)

        # Convert tensor into dict containing indicies of sents to be used as input
        ext_indicator_dict = self.obtain_extraction_indicator_dict(extraction_indicator)

        p, p_mask = self.obtain_extraction_probabilities(h, context, ext_indicator_dict, sent_mask, use_init_embedding)

        return p, p_mask

    def obtain_extraction_probabilities(self, embeddings, context, ext_indicator_dict, sent_mask, use_init_embedding):
        """
        Obtains extraction probabilities per sentence

        :param embeddings:          torch.tensor containing sentence embeddings.
                                    Shape: (batch_size, n_src_sents, emb_dim)
        :param context:             torch.tensor containing self-attention weighted sum of each embedding in input
                                    embeddings. Shape: (batch_size, n_src_sents, context_dim)
        :param ext_indicator_dict:  dictionary of lists in which each entry indicates if the sent embedding
                                    should be used as input to calculate the NEXT sentence to extract.
        :param sent_mask:           A torch.tensor (bool) indicating if the embedding actually exists in the source
                                    doc. Necessary because operating in batches. Shape: (batch_size, n_src_sents)
        :param use_init_embedding:  A bool indicating whether or not the initial embedding should be used
        :return:
          - p:  A torch.tensor containing the extraction probability of each sentence.
                Shape: (batch_size, n_ext_sents+1, n_src_sents)
          - p_mask: A torch.tensor (bool) indicating if the sentence extraction probabilities should be considered.
                    Necessary because of batches. Not all documents have an equal amount of sentences being extracted.
                    Shape: (batch_size, n_ext_sents+1)
        """
        n_batches = embeddings.shape[0]
        p = list()
        for batch_idx in range(n_batches):
            # Obtain appropriate sentence embeddings
            selected_context = self.select_embeddings(
                embeddings=context[batch_idx],
                indicies=ext_indicator_dict.get(batch_idx, None),
                use_init_embedding=use_init_embedding
            )

            # Calculate self attention values
            u = self.linear_h(embeddings[batch_idx]) + self.linear_e(selected_context).unsqueeze(1)
            u = self.tanh(u)
            u = self.linear_v(u)
            u = u.view(-1, u.shape[1])

            # Transform into binary probabilities
            temp_p = self.sigmoid(u) * sent_mask[batch_idx]

            # Collect probs per batch
            p.append(temp_p)

        # Combine probs into one torch.tensor
        p = torch.nn.utils.rnn.pad_sequence(p, batch_first=True, padding_value=self.padding_value)

        # Form mask. Required because not all documents in batch have the same amount of sentence extracted.
        p_mask = ~((p == self.padding_value).sum(dim=2) >= p.shape[-1])
        if p.shape[1] > 1:
            p_mask[range(p_mask.shape[0]), (p_mask.sum(dim=1) - 1).tolist()] = False

        return p, p_mask

    def select_embeddings(self, embeddings, indicies, use_init_embedding):
        """
        Selects tensors appropriately amongst input embeddings. Adds initial embedding if indicated

        :param embeddings: torch.tensor to select embeddings from
        :param indicies: indicies indicating tensors to select
        :param use_init_embedding: boolean indicating whether or not to add the initial sentence embedding
        :return: A torch.tensor containing appropriate embeddings
        """
        if indicies is not None:
            selected_context = embeddings[indicies]
        else:
            selected_context = torch.tensor([])
        if use_init_embedding:
            selected_context = torch.cat([self.init_sent_embedding, selected_context])

        return selected_context

    @staticmethod
    def obtain_extraction_indicator_dict(extraction_indicator):
        """
        Converts torch.tensor into a dictionary of indicies of sents

        :param extraction_indicator: None or torch.tensor of shape: (n_documents, n_src_sents)
        :return: A dictionary of lists indexed by batch_index. Each entry in the list indicates if the sent embedding
                 should be used as input to calculate the NEXT sentence to extract.
        """
        ext_indicator_dict = {}
        if extraction_indicator is not None:
            ext_indicator_dict = defaultdict(list)
            batch_indicies, sent_indicies = torch.where(extraction_indicator)
            for row, pos in zip(batch_indicies, sent_indicies):
                ext_indicator_dict[row.item()].append(pos.item())
        return ext_indicator_dict


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
