from bert.utils import BERT_OUTPUT_SIZE
from collections import defaultdict
from torch import nn
from utils import compute_rouge_l
from utils import MaskedSoftmax

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
            bidirectional=True,
            batch_first=True
        )

        self.ptr_lstm = nn.LSTM(
            input_size=BI_LSTM_OUTPUT_SIZE * 2,
            hidden_size=BI_LSTM_OUTPUT_SIZE * 2,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )

        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

        self.freeze_weights(self.bert_model)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.masked_softmax = MaskedSoftmax(dim=-1)

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
          - extraction_prob:  torch.tensor of shape (batch_size, n_ext_sents, n_src_sents) containing probability of
                              extracting each sentence individually (not a distribution, just binary)
          - extraction_mask:  torch.tensor of shape (batch_size, n_ext_sents) indicating if entries in 'extraction_prob'
                              should be considered legitmate. Necessary because operating in batches.
        """
        # Obtain hidden states for each sentence embedding from bi-RNN
        bi_lstm_hidden_state, __ = self.bi_lstm(sent_embeddings)
        bi_lstm_hidden_state, __ = torch.nn.utils.rnn.pad_packed_sequence(bi_lstm_hidden_state, batch_first=True)

        # Convert tensor into dict containing indicies of sents to be used as input
        ext_indicator_dict = self.obtain_extraction_indicator_dict(extraction_indicator)

        extraction_prob, extraction_mask = self.obtain_extraction_probabilities(
            bi_lstm_hidden_state=bi_lstm_hidden_state,
            ext_indicator_dict=ext_indicator_dict,
            src_sent_mask=sent_mask,
            use_init_embedding=use_init_embedding
        )

        return extraction_prob, extraction_mask

    def obtain_extraction_probabilities(self, bi_lstm_hidden_state, ext_indicator_dict, src_sent_mask, use_init_embedding):
        """
        Obtains extraction probabilities per sentence
        :param bi_lstm_hidden_state: torch.tensor containing hidden embeddings from bidirectional RNN
        :param ext_indicator_dict:  dict of lists in which each entry indicates if the sent embedding should be used as
                                    input to calculate the next sentence to extract.
        :param src_sent_mask:       A torch.tensor (bool) indicating if the embedding actually exists in the source
                                    doc. Necessary because operating in batches. Shape: (batch_size, n_src_sents)
        :param use_init_embedding:  A bool indicating whether or not the initial embedding should be used
        :return:
          - p:  A torch.tensor containing the extraction probability of each sentence.
                Shape: (batch_size, n_ext_sents+1, n_src_sents)
          - extraction_mask: A torch.tensor (bool) indicating if the sentence extraction probabilities should be
                             considered. Necessary because of batches. Not all documents have an equal amount of
                             sentences being extracted. Shape: (batch_size, n_ext_sents+1) (+1 because we still predict
                             next sentence for last sentence extracted. This is masked out in extraction_mask though)
        """
        # Obtain pointer LSTM hidden states
        input_sent_embeddings, n_ext_sents = self.obtain_pointer_network_inputs(
            src_doc_sent_embeddings=bi_lstm_hidden_state,
            ext_indicator_dict=ext_indicator_dict,
            use_init_embedding=use_init_embedding
        )
        ptr_hidden_state, __ = self.ptr_lstm(input_sent_embeddings)
        ptr_hidden_state, __ = torch.nn.utils.rnn.pad_packed_sequence(ptr_hidden_state, batch_first=True)

        # Self attention: "Glimpse", using dot attention mechanism
        attn_values = torch.bmm(ptr_hidden_state, bi_lstm_hidden_state.transpose(1, 2))
        self_attn_weights = self.softmax(attn_values)
        context = torch.bmm(self_attn_weights, bi_lstm_hidden_state)

        # Calculate self attention values again ("hop attention")
        attn_values = torch.bmm(context, bi_lstm_hidden_state.transpose(1, 2))

        # Formulate masks
        src_doc_mask, extraction_mask = self.obtain_masks(src_sent_mask, ext_indicator_dict, attn_values, n_ext_sents)

        # Obtain extraction probabilities
        extraction_prob = self.masked_softmax(attn_values, src_doc_mask)
        if extraction_prob.shape[1] > 1:
            extraction_mask[range(extraction_mask.shape[0]), (extraction_mask.sum(dim=1) - 1).tolist()] = False

        return extraction_prob, extraction_mask

    @staticmethod
    def obtain_masks(src_sent_mask, ext_indicator_dict, attn_values, n_ext_sents):
        """
        :param src_sent_mask:
        :param ext_indicator_dict:
        :param attn_values:
        :param n_ext_sents:
        :return:
        """
        n_src_sents = src_sent_mask.sum(dim=1).int()
        src_doc_mask = torch.ones(attn_values.shape)
        extraction_mask = torch.ones(attn_values.shape[:2]).bool()

        for idx, n_sents in enumerate(n_ext_sents):
            src_doc_mask[idx, n_sents:, :] = 0
            extraction_mask[idx, n_sents:] = False
            # Also src_doc_mask out things that have already been extracted
        for idx, n_sents in enumerate(n_src_sents):
            src_doc_mask[idx, :, n_sents:] = 0
        for batch_idx, ext_sent_indices in ext_indicator_dict.items():
            # Don't extract the same thing again
            for idx, __ in enumerate(ext_sent_indices):
                src_doc_mask[batch_idx, idx+1, [ext_sent_indices[:idx+1]]] = 0

        return src_doc_mask, extraction_mask

    def obtain_pointer_network_inputs(self, src_doc_sent_embeddings, ext_indicator_dict, use_init_embedding):
        """
        :param src_doc_sent_embeddings:
        :param ext_indicator_dict:
        :param use_init_embedding:
        :return:
        """
        n_batches = src_doc_sent_embeddings.shape[0]
        h_selected = torch.cat([self.init_sent_embedding] * n_batches).unsqueeze(1)
        n_ext_sents = torch.ones(n_batches).int()
        if ext_indicator_dict:
            h_selected = list()
            for i in range(n_batches):
                temp_h = src_doc_sent_embeddings[i, ext_indicator_dict[i]]
                if use_init_embedding:
                    temp_h = torch.cat([self.init_sent_embedding, temp_h])
                h_selected.append(temp_h)
            h_selected = torch.nn.utils.rnn.pad_sequence(h_selected, batch_first=True)
            n_ext_sents = [len(x) + 1 for x in ext_indicator_dict.values()]

        h_selected = torch.nn.utils.rnn.pack_padded_sequence(
            h_selected,
            lengths=n_ext_sents,
            batch_first=True,
            enforce_sorted=False
        )

        return h_selected, n_ext_sents

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
