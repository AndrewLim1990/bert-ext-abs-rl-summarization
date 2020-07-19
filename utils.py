import torch


def euclidean_distance(vector_1, vector_2):
    dist = torch.sqrt(torch.sum(vector_1 - vector_2)**2)
    return dist
