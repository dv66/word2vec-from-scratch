import word2vec_dataset
import torch
import pickle
import numpy as np
from word2vec_dataset import Word2VecDataset
from word2vec_model import SkipGramDataset


model = torch.load('../out/word2vec_trained-small-2.pt')
word2vec = word2vec_dataset.Word2VecDataset('../out/backup/sentences-small-2.txt.pkl', is_pickle=True)
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
    word_1 = 'বংশোদ্ভূত'
    word_2 = 'রাজা'

    rand_word_idx = np.random.randint(0, len(word2vec.distinct_words), 30)
    for r in rand_word_idx:
        reference_word = word2vec.distinct_words[r]
        print(f"reference : {reference_word}")
        print(get_k_most_similar_words(reference_word, k=10))
    # cosine_similarity = torch.nn.CosineSimilarity(dim=-1)(get_word_vector(word_1), get_word_vector(word_2)).item()
    # print(f"similarity = {cosine_similarity}")


