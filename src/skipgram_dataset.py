import os

from word2vec_dataset import Word2VecDataset
import linecache
import torch

class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, word2vec: Word2VecDataset, window_size: int = 3):
        self._word2vec = word2vec
        self._target_context_file = open(f'{word2vec.get_file().name}_target_context.txt', 'w')
        self._target_context_filename = self._target_context_file.name
        word2vec.generate_target_context_pairs(
            window=window_size,
            input_file=word2vec.get_file(),
            output_file=self._target_context_file
        )
        self._target_context_file = open(f'{word2vec.get_file().name}_target_context.txt')
        self._total_data = len(self._target_context_file.readlines())

    def __getitem__(self, index):
        line = linecache.getline(self._target_context_filename, index + 1)
        X, y = line.strip().split()
        X_index = self._word2vec.word_index_dict[X]
        y_index = self._word2vec.word_index_dict[y]
        return X_index, y_index

    def __len__(self):
        return self._total_data



