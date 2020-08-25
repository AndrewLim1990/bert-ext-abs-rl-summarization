from torch.nn.utils.rnn import pad_sequence

import numpy as np
import torch

BERT_OUTPUT_SIZE = 768
PADDING_VALUE = -1
START_OF_SENTENCE_TOKEN = "[CLS]"


def obtain_word_embeddings(model, tokenizer, input_documents, static_embeddings=False):
    """
    :param model: bert model to convert word index to word embedding
    :param tokenizer: bert tokenizer to convert word to word index
    :param input_documents: list of sentence strings
    :param static_embeddings:
    :return:
    """
    start_idx = torch.tensor([tokenizer.cls_token_id])
    end_idx = torch.tensor([tokenizer.sep_token_id])

    # Tokenized document sentences and prepend/append on start/end tokens
    documents = list()
    for document in input_documents:
        if document:
            tokenized_document = torch.cat(
                [torch.tensor(tokenizer.encode(input_sentence)) for input_sentence in document]
            )
            tokenized_document = torch.cat([start_idx, tokenized_document, end_idx])
        else:
            tokenized_document = torch.cat([start_idx, end_idx])
        documents.append(tokenized_document)

    doc_lengths = [len(doc) - 1 for doc in documents]  # minus 1 because we don't generate prediction for [SEP] token
    documents = torch.nn.utils.rnn.pad_sequence(documents).T

    if static_embeddings:
        last_hidden_state = torch.cat([model(doc.view(-1, 1))[0].transpose(0, 1) for doc in documents])
    else:
        last_hidden_state, pooler_output = model(documents)
    word_embeddings = last_hidden_state

    mask = torch.ones(word_embeddings.shape)

    for idx, doc_length in enumerate(doc_lengths):
        mask[idx, doc_length:] = 0

    return word_embeddings, mask, documents


def obtain_sentence_embedding(model, tokenizer, input_sentence):
    """
    :param model:
    :param tokenizer:
    :param input_sentence:
    :return:
    """
    input_sentence = torch.tensor(
        tokenizer.encode(f"{START_OF_SENTENCE_TOKEN} {input_sentence}")
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
