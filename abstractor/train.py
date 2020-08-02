import numpy as np
import torch

from bert.utils import obtain_word_embeddings

nll_loss = torch.nn.NLLLoss(reduction='none')


def train_abstractor(
        model, data, learning_rate=1e-3, n_iters=10000, model_output_file="results/models/abstractor.pt", save_freq=2
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(n_iters):
        # Todo: Get more than two
        source_documents, target_summaries = get_training_batch(data, 2)

        # Obtain embeddings
        source_document_embeddings, __, __ = obtain_word_embeddings(
            model.bert_model, model.bert_tokenizer, source_documents
        )
        target_summary_embeddings, target_mask, target_tokens = obtain_word_embeddings(
            model.bert_model, model.bert_tokenizer, target_summaries
        )

        # Shift target tokens and format masks
        target_mask = torch.flatten(target_mask[:, :, 0])
        target_tokens = torch.roll(target_tokens, dims=1, shifts=-1)  # shift left
        target_tokens[:, -1] = 0
        target_tokens = torch.flatten(target_tokens)

        # Obtain extraction probability for each word in vocabulary
        extraction_probabilities, teacher_forcing = model(
            source_document_embeddings,
            target_summary_embeddings,
            teacher_forcing_pct=0
        )  # (batch_size, n_target_words, vocab_size)

        # Obtain negative log likelihood loss
        # Todo: Double check to see if ordering is correct after flatten and .view() reshaping
        loss = nll_loss(extraction_probabilities.view(-1, model.vocab_size), target_tokens)

        loss = loss * target_mask
        loss = loss.sum()
        print(f"Loss: {loss} (teacher_forcing: {teacher_forcing})")

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % save_freq == 0:
            torch.save(model.state_dict(), model_output_file)

    return


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
