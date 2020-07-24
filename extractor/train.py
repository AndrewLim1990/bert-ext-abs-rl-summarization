import torch

bce_loss = torch.nn.BCELoss(reduction='none')


def train_extractor(model, mask, sentence_embeddings, extraction_labels, learning_rate=1e-3):
    """
    :param model:
    :param mask:
    :param sentence_embeddings:
    :param extraction_labels:
    :param learning_rate:
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    extraction_probabilities = model(sentence_embeddings)

    loss = bce_loss(input=extraction_probabilities, target=extraction_labels)
    loss = loss * mask
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return extraction_probabilities
