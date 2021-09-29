import functools
import collections
import random


def negative_sampling_preprocess(words):
    global unigram_table
    global unique_words
    total_words = len(words)
    unique_words = list(set(words))
    word_freq = collections.defaultdict(lambda: 0)
    word_indices = {unique_words[i]: i for i in range(len(unique_words))}

    for word in words:
        word_freq[word] += 1

    alpha = 3/4 # smoothing parameter
    word_prob = {word: (word_freq[word]/total_words) ** alpha for word in unique_words}

    unigram_table = [[i] * round(word_prob[unique_words[i]] * total_words) for i in range(len(unique_words))]
    unigram_table = [i for arr in unigram_table for i in arr]

def get_negative_sample():
    negative_sample_index = random.choice(unigram_table)
    return unique_words[negative_sample_index]

if __name__ == '__main__':
    negative_sampling_preprocess(['a', 'a', 'a', 'b', 'c', 'c', 'c', 'c'])
    print(get_negative_sample())
