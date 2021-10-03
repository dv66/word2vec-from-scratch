import numpy as np

from word2vec_dataset import Word2VecDataset
from skipgram_dataset import SkipGramDataset
import torch

if __name__ == '__main__':
    # wv = Word2VecDataset('../out/test.txt')
    #
    # skpdata = SkipGramDataset(wv, 4)
    #
    # for i in range(len(skpdata)):
    #     print(skpdata[i])
    t1 = torch.Tensor([1,3,7,1])

    for i in range(len(t1)):

        print(int(t1[i].item()))
