from bert.utils import bert_model
from bert.utils import bert_tokenizer
from bert.utils import obtain_sentence_embeddings

import torch

bce_loss = torch.nn.BCELoss(reduction='none')


def train_extractor(model, data, learning_rate=1e-3, n_iters=1000):
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
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(n_iters):
        documents, extraction_labels = get_training_batch(data, batch_size=5)

        sentence_embeddings, mask = obtain_sentence_embeddings(bert_model, bert_tokenizer, documents)

        # Predict probability of extraction per sentence
        extraction_probabilities = model(sentence_embeddings)

        # Calculate loss
        loss = bce_loss(input=extraction_probabilities, target=extraction_labels)
        loss = loss * mask
        loss = loss.sum()
        print(f"Loss: {loss}")

        # Calculate gradient and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


def get_training_batch(training_dictionaries, batch_size):
    """
    Todo: Randmize batches obtained
    :param training_dictionaries:
    :param batch_size:
    :return:
    """

    mini_batch = training_dictionaries[:batch_size]

    documents, extraction_labels = map(list, zip(*[(s['document'], s['extraction_label']) for s in mini_batch]))
    extraction_labels = torch.nn.utils.rnn.pad_sequence(
        sequences=extraction_labels,
        batch_first=True,
        padding_value=0.0
    )

    return documents, extraction_labels
