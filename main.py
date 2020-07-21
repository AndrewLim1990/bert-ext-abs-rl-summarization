from bert.utils import bert_model
from bert.utils import bert_tokenizer
from bert.utils import obtain_sentence_embeddings
# from extractor.train import train_extractor
from extractor.utils import bidirectional_lstm
from extractor.utils import pointer_lstm
from extractor.utils import ExtractorModel
from utils import euclidean_distance

import pickle


def main():
    documents = pickle.load(open('data/sampled_documents.pkl', 'rb'))
    summaries = pickle.load(open('data/sampled_summaries.pkl', 'rb'))

    # Obtain sentence embeddings of shape: (n_docs, max_seq_len, bert_hidden_len)
    sentence_embeddings = obtain_sentence_embeddings(bert_model, bert_tokenizer, documents)

    for sentence, embedding in zip(documents[0], sentence_embeddings[0]):
        dist = euclidean_distance(sentence_embeddings[0][0], embedding)
        print(f"'{documents[0][0]}' to '{sentence}' distance: {dist}")

    print("\n")

    extractor_model = ExtractorModel(bidirectional_lstm, pointer_lstm)
    # train_extractor(extractor_model, documents=documents, summaries=summaries)
    print("Extraction probability distribution amongst sentences: {}".format(extractor_model(sentence_embeddings)))


if __name__ == '__main__':
    main()
