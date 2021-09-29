import random

import numpy as np
import collections


class Word2VecDataset:

    def __init__(self, file_path):
        self.file = open(file_path)
        self.distinct_words = self.get_distinct_words(self.file)
        self.vocabulary_size = len(self.distinct_words)
        self.word_index_dict = dict([(self.distinct_words[i], i) for i in range(len(self.distinct_words))])
        self.unigram_table = self.get_unigram_table_neg_sampling()

    @staticmethod
    def get_distinct_words(file):
        words = []
        for sentence in file:
            for word in sentence.strip().split():
                words.append(word)
        words = list(set(words))

        return words

    def get_unigram_table_neg_sampling(self):
        total_words = len(self.distinct_words)
        word_freq = collections.defaultdict(lambda: 0)
        word_indices = {self.distinct_words[i]: i for i in range(len(self.distinct_words))}

        for word in self.distinct_words:
            word_freq[word] += 1

        alpha = 3 / 4  # smoothing parameter
        word_prob = {word: (word_freq[word] / total_words) ** alpha for word in self.distinct_words}

        unigram_table = [[i] * round(word_prob[self.distinct_words[i]] * total_words) for i in
                         range(len(self.distinct_words))]
        unigram_table = [i for arr in unigram_table for i in arr]

        return unigram_table

    def get_negative_sample(self):
        negative_sample_index = random.choice(self.unigram_table)
        return self.distinct_words[negative_sample_index]

    def get_k_negative_samples(self, k, input_word):
        k_samples = []
        for i in range(k):
            k_samples.append(
                (self.get_one_hot_vector([input_word]), self.get_one_hot_vector([self.get_negative_sample()]))
            )
        return k_samples


    def get_one_hot_vector(self, wordlist: list):
        vector = np.zeros(self.vocabulary_size)
        for word in wordlist:
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
    def generate_target_context_pairs(window, input_file_path, output_file_path):
        output_pairs = []
        for sentence in open(input_file_path):
            sentence = sentence.strip()
            pairs_str = [(w[0] + " " + w[1]) for w in Word2VecDataset.get_target_context_pairs(window, sentence)]
            for pair in pairs_str:
                output_pairs.append(pair)

        open(output_file_path, 'w').write('\n'.join(output_pairs))


if __name__ == '__main__':
    word2vec = Word2VecDataset('../sentences-small.txt')

    print(word2vec.get_k_negative_samples(5, 'মানুষেরা'))
    # Word2VecDataset.generate_target_context_pairs(3, '../sentences-small.txt', '../target-context.txt')
    # [print(x) for x in Word2VecDataset.get_target_context_pairs(3, 'অনেক সময় সহজ জিনিসগুলো নার্ভাসনেসের কারণে ভুলে যাই')]
