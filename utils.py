from torch import nn

import torch


def euclidean_distance(vector_1, vector_2):
    dist = torch.sqrt(torch.sum(vector_1 - vector_2)**2)
    return dist


# Copied from: https://github.com/ChenRocks/fast_abs_rl/tree/b1b66c180a135905574fdb9e80a6da2cabf7f15c
def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp


# Copied from: https://github.com/ChenRocks/fast_abs_rl/tree/b1b66c180a135905574fdb9e80a6da2cabf7f15c
def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]


# Copied from: https://github.com/ChenRocks/fast_abs_rl/tree/b1b66c180a135905574fdb9e80a6da2cabf7f15c
def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def batched_index_select(input, dim, index):
    """
    :param input:
    :param dim:
    :param index:
    :return:
    """
    input = torch.nn.utils.rnn.pad_packed_sequence(input, batch_first=True)[0]
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.reshape(views).expand(expanse)
    return torch.gather(input, dim, index)


def logit(x):
    """
    Applies a logit function robust to x=0
    :param x: float value satisfying: 0 <= x < 1
    :return: the logit of input x
    """
    epsilon = torch.tensor(1e-16)
    x = torch.min(1 - (epsilon * 1e10), torch.max(epsilon, x))
    z = torch.log(x / (1 - x))
    return z


class MaskedSoftmax(nn.Module):
    def __init__(self, dim=0):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        epsilon = torch.tensor([1e-8])
        x_max = x.max(-1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            mask = mask.float()
            x_exp = x_exp * mask.float()
        dist = x_exp / torch.max(x_exp.sum(-1), epsilon).unsqueeze(-1)
        return dist
