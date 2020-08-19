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
    stop_action_index = source_sentence_embeddings.shape[1]
    target_summary_embeddings, target_mask, target_tokens = obtain_word_embeddings(
        rl_model.abstractor_model.bert_model,
        rl_model.abstractor_model.bert_tokenizer,
        target_summaries,
        static_embeddings=True
    )

    for i in range(n_iters):
        # Run trajectory
        actions, log_probs, values = rl_model.sample_actions(source_sentence_embeddings, source_mask)

        # Obtain abstracted sentence from abstractor
        abstract_sentence_indicies = rl_model.create_abstracted_sentences(
            actions,
            source_documents,
            stop_action_index,
            teacher_forcing_pct=1.0,
            target_summary_embeddings=target_summary_embeddings
        )

        # Obtain returns from ROUGE
        rewards = rl_model.determine_rewards(actions, abstract_sentence_indicies, target_tokens, target_mask)

        # Update RL model
        last_action_mask = rl_model.last_action_mask(actions)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        trajectory = (actions, rewards, log_probs, values, last_action_mask)
        rl_model.update(
            state=source_sentence_embeddings,
            trajectory=trajectory
        )


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

