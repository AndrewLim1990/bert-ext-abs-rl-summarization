import numpy as np
import torch

from bert.utils import obtain_word_embeddings
from abstractor.utils import obtain_initial_hidden_states


def train_abstractor(
        model, data, learning_rate=1e-3, n_iters=10000, model_output_file="results/models/extractor.pt", save_freq=2
):
    source_documents, target_summaries = get_training_batch(data, 2)

    # Obtain embeddings
    source_document_embeddings, source_mask = obtain_word_embeddings(
        model.bert_model, model.bert_tokenizer, source_documents
    )
    target_summary_embeddings, target_mask = obtain_word_embeddings(
        model.bert_model, model.bert_tokenizer, target_summaries
    )

    initial_hidden_states = obtain_initial_hidden_states(source_document_embeddings, source_mask)

    # Obtain extraction probability for each word in vocabulary
    extraction_probabilities = model(
        source_document_embeddings,
        target_summary_embeddings,
        initial_hidden_states
    )  # (batch_size, n_target_words, vocab_size)

    # Obtain negative log likelihood loss

    # Update model


    return source_documents


def get_training_batch(training_dictionaries, batch_size):
    """
    :param training_dictionaries:
    :param batch_size:
    :return:
    """
    # Todo: Use randomized batches
    mini_batch = training_dictionaries[:batch_size]  # np.random.choice(training_dictionaries, batch_size).tolist()

    documents, extraction_labels, summaries = map(
        list,
        zip(*[(s['document'], s['extraction_label'], s['summary']) for s in mini_batch])
    )

    extracted_documents = [np.array(documents[i])[extraction_labels[i].bool()].tolist() for i in range(len(documents))]

    return extracted_documents, summaries
