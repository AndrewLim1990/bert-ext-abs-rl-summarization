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
    :return: tuple (cls_embeddings, mask)
        where cls_embeddings: torch.tensor of shape (n_docs, n_sentences_max, bert_embedding_dim)
        where mask: torch.tensor of shape (n_docs, n_sentences_max) This masks out sentences in documents that don't
                    have the maximum number of sentences within the batch.
    """
    n_docs = len(documents)
    cls_embeddings = [torch.cat([
        obtain_sentence_embedding(model, tokenizer, sentence) for sentence in doc
    ]) for doc in documents]  # list w/ length=n_docs and items w/ shape=(n_sentences, dim_bert))

    max_sentence_length = max([x.shape[0] for x in cls_embeddings])
    mask = torch.ones((n_docs, max_sentence_length))

    for idx, cls_embedding in enumerate(cls_embeddings):
        mask[idx, len(cls_embedding):] = 0

    cls_embeddings = pad_sequence(cls_embeddings, padding_value=PADDING_VALUE, batch_first=True)

    return cls_embeddings, mask


def obtain_sentence_embeddings_siamese(model, input_sentences):
    """
    :param model:
    :param input_sentences:
    :return:
    """
    encoded_sentences = np.stack(model.encode(input_sentences))
    encoded_sentences = torch.tensor(encoded_sentences)

    return encoded_sentences
