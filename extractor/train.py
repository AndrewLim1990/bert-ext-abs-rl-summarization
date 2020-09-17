from bert.utils import obtain_sentence_embeddings
from utils import logit

import numpy as np
import torch

cross_entropy_loss = torch.nn.CrossEntropyLoss()


def train_extractor(
        ext_model, data, learning_rate=1e-2, n_iters=10000, model_output_file="results/models/extractor.pt", save_freq=2
):
    """
    Train a model to extract sentences from documents to be used to form summaries.

    :param ext_model: An ExtractorModel to train
    :param data: list of dictionaries of form:
        [
            {'summary': [...], 'document': [...], 'extraction_labels': [...]},
            {...},
            ..
        ]
    :param learning_rate: learning rate to use during training
    :param n_iters: number of iterations to train for
    :param model_output_file: Directory to save ExtractorModel
    :param save_freq: An int indicating how often to save the ExtractorModel
    :return:
    """
    optimizer = torch.optim.Adam(ext_model.parameters(), lr=learning_rate)
    losses = list()

    for i in range(n_iters):
        documents, extraction_labels = get_training_batch(data, batch_size=5)

        sent_embeddings, sent_mask = obtain_sentence_embeddings(
            ext_model.bert_model,
            ext_model.bert_tokenizer,
            documents
        )

        # Predict probability of extraction per sentence
        extraction_probabilities, extraction_sent_mask = ext_model.forward(
            sent_embeddings=sent_embeddings,
            sent_mask=sent_mask,
            extraction_indicator=extraction_labels
        )

        # Calculate loss
        targets = torch.where(extraction_labels)[1]
        extraction_probabilities = extraction_probabilities[extraction_sent_mask]
        extraction_logits = logit(extraction_probabilities)
        loss = cross_entropy_loss(input=extraction_logits, target=targets)
        losses.append(loss)
        print(f"Loss: {loss}")

        # Calculate gradient and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % save_freq == 0:
            torch.save(ext_model.state_dict(), model_output_file)

    return


def get_training_batch(training_dictionaries, batch_size):
    """
    Obtains a random batch to train on

    :param training_dictionaries: A list of dictionaries of form:
        [
            {'summary': [...], 'document': [...], 'extraction_labels': [...]},
            {...},
            ..
        ]
    :param batch_size: An int indicating how many training samples to fetch
    :return:
      - documents: A list of lists of strings. Each string within the list is a sentence within the source document
      - extraction labels: A torch.tensor (bool) containing lables for which sentences should be extracted (oracle)
    """
    mini_batch = np.random.choice(training_dictionaries, batch_size).tolist()

    documents, extraction_labels = map(list, zip(*[(s['document'], s['extraction_label']) for s in mini_batch]))
    extraction_labels = torch.nn.utils.rnn.pad_sequence(
        sequences=extraction_labels,
        batch_first=True,
        padding_value=0.0
    )

    return documents, extraction_labels
