from abstractor.train import train_abstractor
from abstractor.utils import AbstractorModel
from extractor.train import train_extractor
from extractor.utils import ExtractorModel
from data.utils import load_training_dictionaries

import torch


def main():
    training_dictionaries = load_training_dictionaries()

    # extractor_model = ExtractorModel()
    #
    # train_extractor(
    #     extractor_model,
    #     data=training_dictionaries
    # )

    abstractor_model = AbstractorModel()

    train_abstractor(
        abstractor_model,
        data=training_dictionaries
    )


if __name__ == '__main__':
    main()
