from pytorch_transformers import BertModel
from pytorch_transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import torch

PADDING_VALUE = -1

def obtain_sentence_embedding(model, tokenizer, input_sentence):
    """
    :param model:
    :param tokenizer:
    :param input_sentence:
    :return:
    """
    input_sentence = torch.tensor(
        tokenizer.encode(f"[CLS] {input_sentence}")
    ).unsqueeze(0)
    last_hidden_state, pooler_output = model(input_sentence)
    last_hidden_state = last_hidden_state.squeeze()
    cls_embedding = last_hidden_state[0].view(1, -1)

    return cls_embedding


def obtain_sentence_embeddings(model, tokenizer, documents):
    """
    # Todo: Make more efficient than two for loops
    :param model:
    :param tokenizer:
    :param documents:
    :return: list of tensors of sentence embeddings per document
    """
    cls_embeddings = pad_sequence([torch.cat([
        obtain_sentence_embedding(model, tokenizer, sentence) for sentence in doc
    ]) for doc in documents], padding_value=PADDING_VALUE, batch_first=True)

    return cls_embeddings


def obtain_sentence_embeddings_siamese(model, input_sentences):
    """
    :param model:
    :param input_sentences:
    :return:
    """
    encoded_sentences = np.stack(model.encode(input_sentences))
    encoded_sentences = torch.tensor(encoded_sentences)

    return encoded_sentences


bert_model_siamese = SentenceTransformer('bert-base-nli-mean-tokens')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')