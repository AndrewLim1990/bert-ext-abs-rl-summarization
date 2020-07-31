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

    @staticmethod
    def freeze_weights(model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, encoder_hidden_states, target_embeddings, initial_hidden_states, teacher_forcing=True):
        """
        Todo: Convert from "dot" attention mechanism to "additive" to match paper
        Todo: Reference for the above Todo: http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture08-nmt.pdf
        Todo: Figure out if should keep unqueezed version.
        Todo: Fix non-teacher-forcing output shape

        Teacher forcing, must use decoder embeddings

        :param encoder_hidden_states: torch.tensor of shape: (batch_size, n_src_words, bert_dim)
        :param target_embeddings:
        :param initial_hidden_states:
        :param teacher_forcing:
        :return: torch.tensor of shape (batch_size, n_target_words, vocab_size)
        """
        batch_size = encoder_hidden_states.shape[0]

        # Initialize input word embeddings
        decoder_input_idx = torch.tensor(self.bert_tokenizer.encode(
            START_OF_SENTENCE_TOKEN
        )).repeat(batch_size).unsqueeze(0)
        decoder_inputs, __ = self.bert_model(decoder_input_idx)
        decoder_inputs = decoder_inputs.view(batch_size, 1, -1)  # (batch_size, 1, bert_dim)

        summary_word_probs = list()

        if teacher_forcing:
            attn_score = torch.bmm(
                target_embeddings,  # (batch_size, n_target_words, bert_dim)
                encoder_hidden_states.transpose(1, 2)  # (batch_size, bert_dim, n_src_words)
            )  # (batch_size, n_target_words, n_src_words)
            attn_weights = F.softmax(attn_score, dim=2)  # (batch_size, n_target_words, n_src_words)

            # Calculate decoder hidden states
            decoder_hidden_states = torch.bmm(
                attn_weights,  # (batch_size, n_target_words, n_src_words)
                encoder_hidden_states  # (batch_size, n_src_words, bert_dim)
            )  # (batch_size, n_target_words, bert_dim)

            # Obtain probability of selecting a word for entire vocab
            # Todo: double check if standard practice to do a log_softmax over ENTIRE VOCAB
            summary_word_probs = F.log_softmax(
                self.linear_out(decoder_hidden_states),
                dim=2
            )  # (batch_size, n_target_words, vocab_size)
        else:
            for i in range(self.max_sequence_length):
                attn_score = torch.bmm(
                    decoder_inputs,
                    encoder_hidden_states.transpose(1, 2)
                )  # (batch_size, 1, n_src_words)
                attn_weights = F.softmax(attn_score, dim=2)  # (batch_size, 1, n_src_words)

                # Calculate decoder hidden states
                decoder_hidden_states = torch.bmm(attn_weights, encoder_hidden_states)  # (batch_size, 1, bert_dim)

                # Obtain probability of selecting a word for entire vocab
                # Todo: double check if standard practice to do a log_softmax over ENTIRE VOCAB
                word_probs = F.log_softmax(self.linear_out(decoder_hidden_states), dim=2)  # (batch_size, 1, vocab_size)

                # Obtain embeddings of words w/ with higest prob
                word_indicies = torch.argmax(word_probs, dim=2)
                decoder_inputs = self.bert_model(word_indicies)[0]  # (batch_size, 1, bert_dim)

                # Record word probability distributions for each time step
                summary_word_probs.append(word_probs)

            # Reformat word probability distributions for each time step
            summary_word_probs = torch.cat(
                summary_word_probs
            ).transpose(0, 1)  # (batch_size, n_summary_words, 1, vocab_size)

        return summary_word_probs


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
