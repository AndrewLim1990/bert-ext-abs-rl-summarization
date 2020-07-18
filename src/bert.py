from pytorch_transformers import BertModel
from pytorch_transformers import BertTokenizer
from sentence_transformers import SentenceTransformer

import numpy as np
import torch


def obtain_sentence_embedding(model, tokenizer, input_sentence):
    input_sentence = torch.tensor(
        tokenizer.encode(f"[CLS] {input_sentence}")
    ).unsqueeze(0)
    last_hidden_state, pooler_output = model(input_sentence)
    last_hidden_state = last_hidden_state.squeeze()
    cls_embedding = last_hidden_state[0].view(1, -1)

    return cls_embedding


def obtain_sentence_embeddings(model, tokenizer, input_sentences):
    cls_embeddings = torch.cat([
        obtain_sentence_embedding(model, tokenizer, s) for s in input_sentences
    ])

    return cls_embeddings


def obtain_sentence_embeddings_siamese(model, input_sentences):
    encoded_sentences = np.stack(model.encode(input_sentences))
    encoded_sentences = torch.tensor(encoded_sentences)

    return encoded_sentences


bert_model_siamese = SentenceTransformer('bert-base-nli-mean-tokens')
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')