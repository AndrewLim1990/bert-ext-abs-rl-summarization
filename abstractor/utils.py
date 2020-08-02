import torch
import torch.nn as nn
import torch.nn.functional as F

from bert.utils import BERT_OUTPUT_SIZE
from bert.utils import START_OF_SENTENCE_TOKEN
from pytorch_transformers import BertModel
from pytorch_transformers import BertTokenizer


class AbstractorModel(nn.Module):
    def __init__(self):
        super(AbstractorModel, self).__init__()

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.freeze_weights(self.bert_model)

        # Todo: Find suitable attention dimension
        self.attn = torch.nn.Linear(BERT_OUTPUT_SIZE * 2, BERT_OUTPUT_SIZE, bias=False)

        self.vocab_size = self.bert_tokenizer.vocab_size
        self.linear_out = torch.nn.Linear(BERT_OUTPUT_SIZE, self.vocab_size)

        self.max_sequence_length = 128
        self.teacher_forcing_pct = 0

    @staticmethod
    def freeze_weights(model):
        for param in model.parameters():
            param.requires_grad = False

    def obtain_word_distribution(self, input_embeddings, encoder_hidden_states):
        attn_score = torch.bmm(
            input_embeddings,  # (batch_size, n_target_words, bert_dim)
            encoder_hidden_states.transpose(1, 2)  # (batch_size, bert_dim, n_src_words)
        )  # (batch_size, n_target_words, n_src_words)
        # attn_score = F.tanh(attn_score)
        attn_weights = F.softmax(attn_score, dim=2)  # (batch_size, n_target_words, n_src_words)

        # Calculate decoder hidden states
        decoder_hidden_states = torch.bmm(
            attn_weights,  # (batch_size, n_target_words, n_src_words)
            encoder_hidden_states  # (batch_size, n_src_words, bert_dim)
        )  # (batch_size, n_target_words, bert_dim)

        # Obtain probability of selecting a word for entire vocab
        summary_word_probs = F.log_softmax(
            self.linear_out(decoder_hidden_states),
            dim=2
        )  # (batch_size, n_target_words, vocab_size)

        return summary_word_probs

    def forward(self, encoder_hidden_states, target_embeddings=None, teacher_forcing_pct=0):
        """
        Todo: Convert from "dot" attention mechanism to "additive" to match paper
        Todo: Reference for the above Todo: http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture08-nmt.pdf

        :param encoder_hidden_states: torch.tensor of shape: (batch_size, n_src_words, bert_dim)
        :param target_embeddings:
        :param teacher_forcing_pct:
        :return: torch.tensor of shape (batch_size, n_target_words, vocab_size)
        """
        batch_size = encoder_hidden_states.shape[0]

        # Determine target_length if not teacher forcing
        target_length = self.max_sequence_length
        if target_embeddings is not None:
            target_length = target_embeddings.shape[1]

        # Initialize input word embeddings
        decoder_input_idx = torch.tensor(self.bert_tokenizer.encode(
            START_OF_SENTENCE_TOKEN
        )).repeat(batch_size).unsqueeze(0).T
        decoder_inputs, __ = self.bert_model(decoder_input_idx)
        decoder_inputs = decoder_inputs.view(batch_size, 1, -1)  # (batch_size, 1, bert_dim)

        summary_word_probs = list()

        # Determine if should teacher force or not
        teacher_forcing = torch.rand(1) <= teacher_forcing_pct

        if teacher_forcing:
            summary_word_probs = self.obtain_word_distribution(
                input_embeddings=target_embeddings,
                encoder_hidden_states=encoder_hidden_states
            )

        else:
            for i in range(target_length):
                word_prob = self.obtain_word_distribution(
                    input_embeddings=decoder_inputs,
                    encoder_hidden_states=encoder_hidden_states
                )

                # Obtain embeddings of words w/ with higest prob
                word_indicies = torch.argmax(word_prob, dim=2)
                decoder_inputs = self.bert_model(word_indicies)[0]  # (batch_size, 1, bert_dim)

                # Record word probability distributions for each time step
                summary_word_probs.append(word_prob)

            # Reformat word probability distributions for each time step
            summary_word_probs = torch.cat(
                summary_word_probs,
                dim=1
            )  # (batch_size, n_summary_words, vocab_size)

        return summary_word_probs, teacher_forcing


def obtain_initial_hidden_states(source_document_embeddings, source_mask):
    """
    Todo: Test this
    :param source_document_embeddings:
    :param source_mask:
    :return:
    """
    batch_indicies = tuple(torch.arange(source_document_embeddings.shape[0]))

    # Obtain indicies of last word
    last_word_indicies = tuple(source_mask[:, :, 0].sum(axis=1).int() - 1)

    # Obtain the word embedding for last words
    intial_hidden_states = source_document_embeddings[batch_indicies, last_word_indicies]

    return intial_hidden_states
