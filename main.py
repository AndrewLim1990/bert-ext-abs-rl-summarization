from bert.utils import bert_model
from bert.utils import bert_tokenizer
from bert.utils import obtain_sentence_embeddings
from utils import euclidean_distance


def main():
    input_sentences = [
        'hello world my name is andrew',
        'hello world my name is jeff',
        'two plus two is four'
    ]
    sentence_embeddings = obtain_sentence_embeddings(bert_model, bert_tokenizer, input_sentences)

    for sentence, embedding in zip(input_sentences, sentence_embeddings):
        dist = euclidean_distance(sentence_embeddings[0], embedding)
        print(f"'{input_sentences[0]}' to '{sentence}' distance: {dist}")


if __name__ == '__main__':
    main()
