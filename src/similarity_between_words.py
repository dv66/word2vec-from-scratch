import torch
import word2vec_dataset
import argparse

parser = argparse.ArgumentParser(description='Print cosine similarity between 2 word vectors.')
parser.add_argument('--model-file', type=str, required=True,
                    help='Trained word2vec model file path')
parser.add_argument('--vocabulary-pickle-file', type=str, required=True,
                    help='Pickle file for vocabulary corresponding to trained model')
parser.add_argument('--reference-word-1', type=str, required=True,
                    help='Reference word #1 for similarity check.')
parser.add_argument('--reference-word-2', type=str, required=True,
                    help='Reference word #2 for similarity check.')

args = parser.parse_args()

model = torch.load(args.model_file)
word2vec = word2vec_dataset.Word2VecDataset(args.vocabulary_pickle_file, is_pickle=True)
distinct_words = word2vec.get_distinct_words()


def get_similarity(w1, w2):
    i1 = distinct_words.index(w1)
    i2 = distinct_words.index(w2)
    v1 = model['hidden.0.weight'][:, i1]
    v2 = model['hidden.0.weight'][:, i2]
    similarity = torch.nn.CosineSimilarity(dim=-1)(v1, v2).item()

    return similarity


if __name__ == '__main__':
    print(get_similarity(args.reference_word_1, args.reference_word_2))


