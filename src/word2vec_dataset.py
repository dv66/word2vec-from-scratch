import random
import pickle
import numpy as np
import collections


class Word2VecDataset:

    def __init__(self, file_path, is_pickle=False):
        self.distinct_words = []
        self.file_path = file_path
        if is_pickle:
            self.distinct_words = pickle.load(open(file_path, 'rb'))['distinct_words']
        else:
            self.file = open(file_path)
            self.extract_distinct_words()
            self.save_word_data(f'{file_path}.pkl')
        self.vocabulary_size = len(self.distinct_words)
        self.word_index_dict = dict([(self.distinct_words[i], i) for i in range(len(self.distinct_words))])
        self.unigram_table = self.get_unigram_table_neg_sampling(smoothing_parameter=3 / 4)

    def get_file(self):
        return open(self.file_path)

    def get_total_distinct_words(self):
        return len(self.distinct_words)

    def get_distinct_words(self):
        return self.distinct_words

    def extract_distinct_words(self):
        for sentence in self.file:
            for word in sentence.strip().split():
                self.distinct_words.append(word)
        self.distinct_words = list(set(self.distinct_words))

    def get_unigram_table_neg_sampling(self, smoothing_parameter=3 / 4):
        total_words = len(self.distinct_words)
        word_freq = collections.defaultdict(lambda: 0)

        for word in self.distinct_words:
            word_freq[word] += 1

        # smoothing parameter = 3/4 according to Mikolov et al.
        word_prob = {word: (word_freq[word] / total_words) ** smoothing_parameter for word in self.distinct_words}

        unigram_table = [[i] * round(word_prob[self.distinct_words[i]] * total_words) for i in
                         range(len(self.distinct_words))]
        unigram_table = [i for arr in unigram_table for i in arr]

        return unigram_table

    def get_negative_sample(self):
        return random.choice(self.unigram_table)

    def get_k_negative_samples(self, k):
        k_samples = [self.get_negative_sample() for i in range(k)]
        return k_samples

    def get_one_hot_vector(self, word: str):
        vector = np.zeros(self.vocabulary_size)
        vector[self.word_index_dict[word]] = 1.0

        return vector.astype(np.float32)

    @staticmethod
    def get_target_context_pairs(window_range: int, sentence: str):
        """
        # from Word2VecDataset Mikolov et al. page 4 :
        C is the maximum distance of the words. Thus, if we choose C = 5, for each training word
        we will select randomly a number R in range < 1; C >, and then use R words from history and
        R words from the future of the current word as correct labels. This will require us to do R × 2
        word classifications, with the current word as input, and each of the R + R words as output.
        :param sentence: A sentence
        :param window_range: number of words left and right from current word
        :return: None
        """
        target_context_pairs = []
        words = sentence.strip().split()

        for i in range(len(words)):
            target = words[i]
            for j in range(max(0, i - window_range), min(i + window_range + 1, len(words))):
                if i == j: continue
                context = words[j]
                target_context_pairs.append((target, context))

        return target_context_pairs

    @staticmethod
    def generate_target_context_pairs(window, input_file, output_file):
        output_pairs = []
        for sentence in input_file:
            sentence = sentence.strip()
            pairs_str = [(w[0] + " " + w[1]) for w in Word2VecDataset.get_target_context_pairs(window, sentence)]
            for pair in pairs_str:
                output_pairs.append(pair)

        random.shuffle(output_pairs)

        output_file.write('\n'.join(output_pairs))


    def save_word_data(self, pickle_file):
        word_data = {
            'distinct_words': self.distinct_words
        }
        with open(pickle_file, 'wb') as f:
            pickle.dump(word_data, f)


if __name__ == '__main__':
    # word2vec = Word2VecDataset('../out/sentences-small-2.txt')
    # word2vec = Word2VecDataset('../out/sentences-small-2.txt.pkl', is_pickle=True)
    # print(len(word2vec.get_distinct_words()))
    # print(word2vec.get_one_hot_vector('স্ন্যাপচ্যাটে'))

    word2vec = Word2VecDataset('../out/test.txt')
    word2vec = Word2VecDataset('../out/test.txt.pkl', is_pickle=True)
    print(word2vec.get_distinct_words())
    word2vec.generate_target_context_pairs(window=3, input_file_path='../out/test.txt',
                                           output_file_path='../out/test-skipgram-data.txt')

    from word2vec_model import SkipGramDataset
    skipgram_data = SkipGramDataset('../out/test-skipgram-data.txt', word2vec)

    for i in range(len(skipgram_data)):
        print(skipgram_data[i])
