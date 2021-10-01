import word2vec_preprocess
import torch
import pickle

model = torch.load('../out/word2vec_trained-small-2.pt')
word2vec = word2vec_preprocess.Word2VecDataset('../out/sentences-small-2.txt.pkl', is_pickle=True)
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
    # word_2 = 'রাজা'

    # print(distinct_words)
    print(get_k_most_similar_words(word_1, k=10))
    # cosine_similarity = torch.nn.CosineSimilarity(dim=-1)(get_word_vector(word_1), get_word_vector(word_2)).item()
    # print(f"similarity = {cosine_similarity}")
