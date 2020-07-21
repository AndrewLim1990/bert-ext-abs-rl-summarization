from bert.utils import bert_model
from bert.utils import bert_tokenizer
from bert.utils import obtain_sentence_embeddings
from extractor.utils import bidirectional_lstm
from extractor.utils import pointer_lstm
from extractor.utils import ExtractorModel
from utils import euclidean_distance


def main():
    input_sentences = [
        'hello world my name is andrew',
        'hello world my name is jeff',
        'two plus two is four'
    ]
    sentence_embeddings = obtain_sentence_embeddings(bert_model, bert_tokenizer, input_sentences)
    sentence_embeddings = sentence_embeddings.unsqueeze(1)  # [n_sentences, n_batches, bert_output_dim]

    # for sentence, embedding in zip(input_sentences, sentence_embeddings):
    #     dist = euclidean_distance(sentence_embeddings[0], embedding)
    #     print(f"'{input_sentences[0]}' to '{sentence}' distance: {dist}")
    #
    # print("\n")

    extractor_model = ExtractorModel(bidirectional_lstm, pointer_lstm)
    print("Result: {}".format(extractor_model(sentence_embeddings)))


if __name__ == '__main__':
    main()
