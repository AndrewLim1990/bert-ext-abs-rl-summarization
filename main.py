from abstractor.train import train_abstractor
from abstractor.utils import AbstractorModelRNN
from data.utils import load_training_dictionaries
from extractor.train import train_extractor
from extractor.utils import ExtractorModel
from pytorch_transformers import BertModel
from pytorch_transformers import BertTokenizer
from rl_connection.utils import RLModel
from rl_connection.train import train_rl

import torch


def main():
    train_new_extractor = False
    train_new_abstractor = False
    continue_rl_training = False

    # torch.autograd.set_detect_anomaly(True)
    training_dictionaries = load_training_dictionaries()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Obtain extractor model
    extractor_model = ExtractorModel(bert_tokenizer, bert_model)
    if train_new_extractor:
        train_extractor(
            extractor_model,
            data=training_dictionaries
        )
    else:
        extractor_model_path = "results/models/extractor.pt"
        extractor_model.load_state_dict(torch.load(extractor_model_path))

    # Obtain abstractor model
    abstractor_model = AbstractorModelRNN(bert_tokenizer, bert_model)
    if train_new_abstractor:
        train_abstractor(
            abstractor_model,
            data=training_dictionaries,
            n_iters=100000
        )
    else:
        abstractor_model_path = "results/models/abstractor.pt"
        abstractor_model.load_state_dict(torch.load(abstractor_model_path))

    rl_model = RLModel(
        extractor_model,
        abstractor_model,
        alpha=1e-3
    )

    if continue_rl_training:
        rl_model.load_state_dict(torch.load("results/models/rl.pt"))

    train_rl(
        rl_model=rl_model,
        data=training_dictionaries
    )


if __name__ == '__main__':
    main()
