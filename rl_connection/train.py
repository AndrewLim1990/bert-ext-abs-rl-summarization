from bert.utils import obtain_word_embeddings
from bert.utils import obtain_sentence_embeddings

import torch
import numpy as np


def train_system(rl_model, data, n_iters=5):
    """
    :param extractor_model:
    :param abstractor_model:
    :param rl_model:
    :param data:
    :return:
    """
    # Todo: Figure out when to create training data: Don't have to get embeddings on each iter
    # Obtain batch:
    source_documents, target_summaries = get_training_batch(data, batch_size=2)

    # Obtain embeddings
    source_sentence_embeddings, source_mask = obtain_sentence_embeddings(
        rl_model.extractor_model.bert_model,
        rl_model.extractor_model.bert_tokenizer,
        source_documents
    )
    target_summary_embeddings, target_mask, target_tokens = obtain_word_embeddings(
        rl_model.abstractor_model.bert_model,
        rl_model.abstractor_model.bert_tokenizer,
        target_summaries,
        static_embeddings=True
    )

    for i in range(n_iters):
        # Things to collect:
        actions = list()
        log_probs = list()
        values = list()

        # Until all sentences are extracted:
        # Todo: Can only do this one sample at a time. Cannot batch because of the stop word thing.
        while True:
            # Obtain action extracted sentence from extractor
            action, log_prob, value = rl_model.sample_action(source_sentence_embeddings)

            # Collect things:
            actions.append(action.unsqueeze(0))
            log_probs.append(log_prob.unsqueeze(0))
            values.append(value.unsqueeze(0))

            # Check if should stop extracting sentences
            # Todo: Check if .any() is appropriate, related to one sample at a time problem above
            n_sentences_extracted = len(actions)
            n_sentences_in_label = target_summary_embeddings.shape[1]
            is_stop_action = action >= source_sentence_embeddings.shape[1]
            is_longer_than_label = n_sentences_extracted >= n_sentences_in_label
            if is_stop_action.any() or is_longer_than_label:
                break

        # Obtain abstracted sentence from abstractor
        # Todo: Obtain word embeddings from extraction step
        # Obtain reward from ROUGE
        # Save trajectory
        # Update RL model
    pass


def get_training_batch(training_dictionaries, batch_size):
    """
    :param training_dictionaries:
    :param batch_size:
    :return:
    """
    # Todo: Use randomized batches
    mini_batch = training_dictionaries[:batch_size]  # np.random.choice(training_dictionaries, batch_size).tolist()

    documents, summaries = map(
        list,
        zip(*[(s['document'], s['summary']) for s in mini_batch])
    )

    return documents, summaries

