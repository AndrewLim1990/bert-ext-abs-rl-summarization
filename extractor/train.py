from bert.utils import obtain_sentence_embeddings

import pickle
import numpy as np
import torch

bce_loss = torch.nn.BCELoss(reduction='none')


def train_extractor(
        model, data, learning_rate=1e-3, n_iters=10000, model_output_file="results/models/extractor.pt", save_freq=2
):
    """
    :param model:
    :param data: list of dictionaries of form:
        [
            {'summary': [...], 'document': [...], 'extraction_labels': [...]},
            {...},
            ..
        ]
    :param learning_rate:
    :param n_iters:
    :param model_output_file:
    :param save_freq:
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = list()

    for i in range(n_iters):
        documents, extraction_labels = get_training_batch(data, batch_size=5)

        sentence_embeddings, mask = obtain_sentence_embeddings(model.bert_model, model.bert_tokenizer, documents)

        # Predict probability of extraction per sentence
        extraction_probabilities = model(sentence_embeddings)

        # Calculate loss
        loss = bce_loss(input=extraction_probabilities, target=extraction_labels)
        loss = loss * mask
        loss = loss.sum()
        losses.append(loss)
        print(f"Loss: {loss}")

        # Calculate gradient and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % save_freq == 0:
            torch.save(model.state_dict(), model_output_file)
            pickle.dump(losses, open("results/models/extractor_losses.pkl", "wb"))

    return


def get_training_batch(training_dictionaries, batch_size):
    """
    :param training_dictionaries:
    :param batch_size:
    :return:
    """
    mini_batch = np.random.choice(training_dictionaries, batch_size).tolist()

    documents, extraction_labels = map(list, zip(*[(s['document'], s['extraction_label']) for s in mini_batch]))
    extraction_labels = torch.nn.utils.rnn.pad_sequence(
        sequences=extraction_labels,
        batch_first=True,
        padding_value=0.0
    )

    return documents, extraction_labels
