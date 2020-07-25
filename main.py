from extractor.train import train_extractor
from extractor.utils import ExtractorModel
from data.utils import load_training_dictionaries

import torch


def main():
    training_dictionaries = load_training_dictionaries()

    extractor_model = ExtractorModel()
    # model_path = "results/models/extractor.pt"
    # extractor_model.load_state_dict(torch.load(model_path))

    train_extractor(
        extractor_model,
        data=training_dictionaries
    )


if __name__ == '__main__':
    main()
