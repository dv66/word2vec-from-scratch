'''
# Sample Usage:

python k_similar_words.py \
--model-file ../frozen-models/word2vec_trained-original_corpus-cleaned-20k.en-3.pt \
--vocabulary-pickle-file ../frozen-models/original_corpus-cleaned-20k.en.pkl \
--reference-word SUCCESS \
--k 20

'''
import torch
import word2vec_dataset
import argparse

parser = argparse.ArgumentParser(description='Print K Similar Words for a given word')
parser.add_argument('--model-file', type=str, required=True,
                    help='Trained word2vec model file path')
parser.add_argument('--vocabulary-pickle-file', type=str, required=True,
                    help='Pickle file for vocabulary corresponding to trained model')
parser.add_argument('--reference-word', type=str, required=True,
                    help='Reference word for similar words')
parser.add_argument('--k', type=int, required=True,
                    help='Number of similar words to print')

args = parser.parse_args()

model = torch.load(args.model_file)
word2vec = word2vec_dataset.Word2VecDataset(args.vocabulary_pickle_file, is_pickle=True)
distinct_words = word2vec.get_distinct_words()

def get_word_vector(word):
    index = distinct_words.index(word)
    return model['hidden.0.weight'][:, index]


def get_k_most_similar_words(word, k):
    word_similarity = []
    index = distinct_words.index(word)
    reference_word_vector = model['hidden.0.weight'][:, index]
    for i in range(len(model['hidden.0.weight'][0])):
        vector = model['hidden.0.weight'][:, i]
        similarity = torch.nn.CosineSimilarity(dim=-1)(reference_word_vector, vector).item()
        word_similarity.append((similarity, i))

    word_similarity = sorted(word_similarity)[::-1][:k]
    print(word_similarity)
    return [distinct_words[x[1]] for x in word_similarity]



if __name__ == '__main__':
    print(get_k_most_similar_words(args.reference_word, k=args.k))
    


