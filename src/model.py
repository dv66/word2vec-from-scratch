'''
# Sample Usage::

python model.py \
--lr=1 \
--word-vector-dimension=300 \
--epochs=15 \
--batch-size=256 \
--model-name=../test-model.pt \
--window-size=5 \
--n-neg-samples=10 \
--gamma=0.7 \
--corpus-file=../original_corpus-cleaned-20k.en
'''


from __future__ import print_function
import argparse
import os
import linecache
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from skipgram_dataset import SkipGramDataset
from word2vec_dataset import Word2VecDataset

writer = SummaryWriter()


class Word2VecNN(nn.Module):
    def __init__(self, input_dimension, word_vec_dimensions):
        super(Word2VecNN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dimension, word_vec_dimensions),
            nn.Linear(word_vec_dimensions, input_dimension)
        )

    def forward(self, x):
        return self.hidden(x)


def skip_gram_loss(data_indices, target_indices, word2vec: Word2VecDataset, model: Word2VecNN, n_neg_samples: int):
    '''
    https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf
    SkipGram objective function by Mikolov et al.
    :param data_indices: indices of center words
    :param target_indices: indices of context words
    :param word2vec: Word2VecDataset object
    :param model: Word2VecNN model
    :param n_neg_samples: number of negative samples
    :return:
    '''
    losses = torch.zeros(len(data_indices)).cuda()
    for i in range(len(data_indices)):
        v_c = model.hidden[0].weight[:, data_indices[i]]
        u = model.hidden[1].weight[target_indices[i]]
        first_term = torch.log2(
            torch.nn.functional.sigmoid(torch.dot(u, v_c)))
        k_negative_samples_indices = word2vec.get_k_negative_samples(k=n_neg_samples)
        second_term = 0.0
        for neg_sample_index in k_negative_samples_indices:
            neg_sample_embedding_vector = model.hidden[1].weight[neg_sample_index]
            second_term += torch.log2(nn.functional.sigmoid(
                - torch.dot(neg_sample_embedding_vector, v_c)))
        losses[i] = - first_term - second_term

    return torch.mean(losses)


def word_index_to_one_hot(indices, word2vec: Word2VecDataset):
    indices = [int(indices[i].item()) for i in range(len(indices))]
    one_hots = torch.Tensor([word2vec.get_one_hot_vector(word2vec.get_distinct_words()[x]) for x in indices])

    return one_hots


def word_similarity_test(model: Word2VecNN, word2vec: Word2VecDataset, num_words, k_similar):
    word_similarity = []
    rand_indices = np.random.uniform(0, len(word2vec.distinct_words), num_words)
    for r in rand_indices:
        reference_word_vector = model.hidden[0][:, r]
        for i in range(len(model.hidden[0][0])):
            vector = model.hidden[0][:, i]
            similarity = torch.nn.CosineSimilarity(dim=-1)(reference_word_vector, vector).item()
            word_similarity.append((similarity, i))
        word_similarity = sorted(word_similarity)[::-1][:k_similar]
        print(f"reference : {word2vec.distinct_words[r]} \n similar words : ")
        print([word2vec.distinct_words[x[1]] for x in word_similarity])
    print("="*50)


def train(args, model, device, train_loader, optimizer, epoch, word2vec, n_neg_samples=10):
    model.train()
    for batch_idx, (data_indices, target_indices) in enumerate(train_loader):
        optimizer.zero_grad()
        # output = model(data)
        loss = skip_gram_loss(data_indices, target_indices, word2vec, model, n_neg_samples)
        writer.add_scalar("Loss/train", loss.item(), epoch)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_indices), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Word2VecDataset Implementation from scratch')
    parser.add_argument('--word-vector-dimension', type=int, default=300, required=True,
                        help='dimension of word embedding')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, required=True,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, required=True,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Filename for saving the current Model')
    parser.add_argument('--window-size', type=int, default=3,
                        help='Window size for creating target context pairs'
                        )
    parser.add_argument('--n-neg-samples', type=int, required=True,
                        help='No. of negative samples against per correct pair'
                        )
    parser.add_argument('--corpus-file', type=str, required=True,
                        help='Text corpus file'
                        )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    word2vec = Word2VecDataset(args.corpus_file)
    dataset = SkipGramDataset(word2vec, window_size=args.window_size)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size
    )

    input_dimension = word2vec.get_total_distinct_words()

    model = Word2VecNN(input_dimension, args.word_vector_dimension).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, data_loader, optimizer, epoch, word2vec, n_neg_samples=args.n_neg_samples)
        torch.save(model.state_dict(), f"snapshot-{epoch}-"+args.model_name)
        scheduler.step()
    writer.flush()

    if args.save_model:
        torch.save(model.state_dict(), f"final-{args.model_name}")


if __name__ == '__main__':
    main()


