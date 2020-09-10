from bert.utils import obtain_word_embeddings
from bert.utils import obtain_sentence_embeddings

import numpy as np
import torch


def train_rl(rl_model, data, n_iters=10000):
    """
    :param rl_model:
    :param data:
    :param n_iters:
    :return:
    """
    for i in range(n_iters):
        # Obtain batch:
        source_documents, target_summaries = get_training_batch(data, batch_size=4)

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

        # Obtain exctracted sentences (actions)
        actions, log_probs, entropys, values, n_ext_sents = rl_model.sample_actions(
            source_sentence_embeddings,
            source_mask
        )

        # Obtain abstracted sentence from abstractor
        predicted_tokens, word_probabilities = rl_model.create_abstracted_sentences(
            actions,
            source_documents,
            n_ext_sents=n_ext_sents,
            teacher_forcing_pct=1.0,
            target_summary_embeddings=target_summary_embeddings
        )

        # Obtain returns from ROUGE
        n_actions = actions.shape[1]
        rewards = rl_model.determine_rewards(n_ext_sents, n_actions, predicted_tokens, target_tokens, target_mask)

        # Calc trajectories
        rl_model.update(
            rewards,
            log_probs,
            entropys,
            values,
            n_ext_sents,
            word_probabilities,
            target_tokens,
            target_mask
        )

        save_freq = 100
        if i % save_freq == 0:
            torch.save(rl_model.state_dict(), "results/models/rl.pt")


def get_training_batch(training_dictionaries, batch_size):
    """
    :param training_dictionaries:
    :param batch_size:
    :return:
    """
    mini_batch = np.random.choice(training_dictionaries, batch_size).tolist()

    documents, summaries = map(
        list,
        zip(*[(s['document'], s['summary']) for s in mini_batch])
    )

    return documents, summaries

