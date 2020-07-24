from bert.utils import bert_model
from bert.utils import bert_tokenizer
from bert.utils import obtain_sentence_embeddings
from extractor.train import train_extractor
from extractor.utils import bidirectional_lstm
from extractor.utils import get_extract_labels
from extractor.utils import pointer_lstm
from extractor.utils import ExtractorModel
from utils import euclidean_distance

import pickle


def main():
    documents = pickle.load(open('data/sampled_documents.pkl', 'rb'))
    summaries = pickle.load(open('data/sampled_summaries.pkl', 'rb'))

    extraction_labels = get_extract_labels(documents, summaries)

    # Obtain sentence embeddings of shape: (n_docs, max_seq_len, bert_hidden_len)
    sentence_embeddings, mask = obtain_sentence_embeddings(bert_model, bert_tokenizer, documents)

    extractor_model = ExtractorModel(bidirectional_lstm, pointer_lstm)

    train_extractor(
        extractor_model,
        mask=mask,
        sentence_embeddings=sentence_embeddings,
        extraction_labels=extraction_labels
    )


if __name__ == '__main__':
    main()
