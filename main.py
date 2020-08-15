from abstractor.train import train_abstractor
from abstractor.utils import AbstractorModelRNN
from data.utils import load_training_dictionaries
from extractor.train import train_extractor
from extractor.utils import ExtractorModel
from pytorch_transformers import BertModel
from pytorch_transformers import BertTokenizer
from rl_connection.utils import RLModel
from rl_connection.train import train_system

import torch


def main():
    training_dictionaries = load_training_dictionaries()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    extractor_model = ExtractorModel(bert_tokenizer, bert_model)

    # train_extractor(
    #     extractor_model,
    #     data=training_dictionaries
    # )

    abstractor_model = AbstractorModelRNN(bert_tokenizer, bert_model)

    # train_abstractor(
    #     abstractor_model,
    #     data=training_dictionaries
    # )

    rl_model = RLModel(
        extractor_model,
        abstractor_model,
    )

    train_system(
        rl_model=rl_model,
        data=training_dictionaries
    )


if __name__ == '__main__':
    main()
