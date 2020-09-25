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
          - p:          torch.tensor of shape (batch_size, n_ext_sents, n_src_sents) containing probability of
                        extracting each sentence individually (not a distribution, just binary)
          - extraction_mask:     torch.tensor of shape (batch_size, n_ext_sents) indicating if entries in 'p' are should be
                        considered legitmate. Necessary because operating in batches
        """
        # Obtain hidden states for each sentence embedding from bi-RNN
        h, __ = self.bi_lstm(sent_embeddings)
        h, __ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        # Convert tensor into dict containing indicies of sents to be used as input
        ext_indicator_dict = self.obtain_extraction_indicator_dict(extraction_indicator)

        p, extraction_mask, src_doc_mask = self.obtain_extraction_probabilities(
            h=h,
            ext_indicator_dict=ext_indicator_dict,
            src_sent_mask=sent_mask,
            use_init_embedding=use_init_embedding
        )

        return p, extraction_mask, src_doc_mask

    def obtain_extraction_probabilities(self, h, ext_indicator_dict, src_sent_mask, use_init_embedding):
        """
        Obtains extraction probabilities per sentence
        :param h:                   torch.tensor containing hidden embeddings from bidirectional RNN
        :param ext_indicator_dict:  dictionary of lists in which each entry indicates if the sent embedding
                                    should be used as input to calculate the NEXT sentence to extract.
        :param src_sent_mask:       A torch.tensor (bool) indicating if the embedding actually exists in the source
                                    doc. Necessary because operating in batches. Shape: (batch_size, n_src_sents)
        :param use_init_embedding:  A bool indicating whether or not the initial embedding should be used
        :return:
          - p:  A torch.tensor containing the extraction probability of each sentence.
                Shape: (batch_size, n_ext_sents+1, n_src_sents)
          - extraction_mask: A torch.tensor (bool) indicating if the sentence extraction probabilities should be considered.
                    Necessary because of batches. Not all documents have an equal amount of sentences being extracted.
                    Shape: (batch_size, n_ext_sents+1) (+1 because we still predict next sentence for last sentence
                    extracted. This is masked out in extraction_mask though)
        """
        n_batches = src_sent_mask.shape[0]
        torch.cat([self.init_sent_embedding] * n_batches).unsqueeze(1)

        h_selected = torch.cat([self.init_sent_embedding] * n_batches).unsqueeze(1)
        n_ext_sents = torch.ones(n_batches).int()
        if ext_indicator_dict:
            h_selected = list()
            for i in range(n_batches):
                temp_h = h[i, ext_indicator_dict[i]]
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
        z, __ = self.ptr_lstm(h_selected)
        z, __ = torch.nn.utils.rnn.pad_packed_sequence(z, batch_first=True)

        # Self attention: "Glimpse", using dot attention mechanism
        attn_values = torch.bmm(z, h.transpose(1, 2))
        self_attn_weights = self.softmax(attn_values)
        context = torch.bmm(self_attn_weights, h)

        # Calculate self attention values again ("hop attention")
        attn_values = torch.bmm(context, h.transpose(1, 2))

        # Formulate masks
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

        p = self.masked_softmax(attn_values, src_doc_mask)

        if p.shape[1] > 1:
            extraction_mask[range(extraction_mask.shape[0]), (extraction_mask.sum(dim=1) - 1).tolist()] = False

        return p, extraction_mask, src_doc_mask

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
