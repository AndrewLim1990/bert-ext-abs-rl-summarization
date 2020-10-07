from bert.utils import obtain_sentence_embeddings
from data.utils import load_training_dictionaries
from utils import logit

import numpy as np
import torch
import pickle

cross_entropy_loss = torch.nn.CrossEntropyLoss()


def train_extractor(
        ext_model,
        data,
        learning_rate=1e-3,
        n_iters=5000000,
        model_output_file="results/models/extractor.pt",
        save_freq=10000
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
    """
    optimizer = torch.optim.Adam(ext_model.parameters(), lr=learning_rate)

    documents, extraction_labels = convert_training_dict(data)
    sent_embeddings, sent_mask = obtain_sentence_embeddings(
        ext_model.bert_model,
        ext_model.bert_tokenizer,
        documents
    )

    # Get validation data
    validation_dictionaries = load_training_dictionaries(
        input_file="data/validation_data/data_dictionaries.pkl"
    )
    val_documents, val_labels = convert_training_dict(validation_dictionaries)
    val_sent_embeddings, val_sent_masks = obtain_sentence_embeddings(
        ext_model.bert_model,
        ext_model.bert_tokenizer,
        val_documents,
        data_dir='data/extractor_data/validation_embeddings/{}',
        load_old=True
    )

    val_doc_lengths = torch.sum(val_sent_masks, dim=1)
    val_sent_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
        val_sent_embeddings,
        lengths=val_doc_lengths,
        batch_first=True,
        enforce_sorted=False
    )

    report_freq = 500
    losses = list()
    for i in range(n_iters):
        # Get random samples
        samp_sent_embeddings, samp_sent_mask, samp_extraction_labels, doc_indicies = get_training_batch(
            sent_embeddings,
            sent_mask,
            extraction_labels,
            batch_size=16
        )

        doc_lengths = torch.sum(samp_sent_mask, dim=1)
        samp_sent_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            samp_sent_embeddings,
            lengths=doc_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # Predict probability of extraction per sentence (training)
        extraction_probabilities, extraction_sent_mask = ext_model.forward(
            sent_embeddings=samp_sent_embeddings,
            sent_mask=samp_sent_mask,
            extraction_indicator=samp_extraction_labels
        )

        # Calculate loss
        training_loss = calc_loss(
            predictions=extraction_probabilities,
            mask=extraction_sent_mask,
            labels=samp_extraction_labels
        )

        if i % report_freq == 0:
            # Predict probability of extraction per sentence (validation)
            val_extraction_probabilities, val_extraction_sent_mask = ext_model.forward(
                sent_embeddings=val_sent_embeddings,
                sent_mask=val_sent_masks,
                extraction_indicator=val_labels
            )
            validation_loss = calc_loss(
                predictions=val_extraction_probabilities,
                mask=val_extraction_sent_mask,
                labels=val_labels
            )
            print(f"Training loss: {training_loss}")
            print(f"Validation loss: {validation_loss}\n")
            losses.append(training_loss)

        # Calculate gradient and update
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
        # if i % save_freq == 0:
        #     torch.save(ext_model.state_dict(), model_output_file)
        #     pickle.dump(losses, open('results/losses/extractor_training_losses.pkl', 'wb'))


def calc_loss(predictions, mask, labels):
    targets = torch.where(labels)[1]
    extraction_probabilities = predictions[mask]
    extraction_logits = logit(extraction_probabilities)
    loss = cross_entropy_loss(input=extraction_logits, target=targets)
    return loss


def get_training_batch(embeddings, masks, labels, batch_size=5):
    """
    Obtains a random set of embeddings, masks, and labels
    """
    n_docs = embeddings.shape[0]
    mini_batch_indicies = np.random.choice(np.arange(n_docs), batch_size)

    samp_embeddings = embeddings[mini_batch_indicies]
    samp_masks = masks[mini_batch_indicies]
    samp_labels = labels[mini_batch_indicies]

    return samp_embeddings, samp_masks, samp_labels, mini_batch_indicies


def convert_training_dict(training_dictionaries):
    """
    Converts training dictionaries to separate documents and extraction_labels

    :param training_dictionaries: A list of dictionaries of form:
        [
            {'summary': [...], 'document': [...], 'extraction_labels': [...]},
            {...},
            ..
        ]
    :return:
      - documents: A list of lists of strings. Each string within the list is a sentence within the source document
      - extraction labels: A torch.tensor (bool) containing lables for which sentences should be extracted (oracle)
    """
    documents, extraction_labels = map(
        list,
        zip(*[(s['document'], s['extraction_label']) for s in training_dictionaries])
    )
    extraction_labels = torch.nn.utils.rnn.pad_sequence(
        sequences=extraction_labels,
        batch_first=True,
        padding_value=0.0
    )

    return documents, extraction_labels
