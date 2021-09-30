from __future__ import print_function
import argparse
import linecache
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from word2vec_preprocess import Word2VecDataset

writer = SummaryWriter()

class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, word2vec: Word2VecDataset):
        self._file_name = file_name
        self._word2vec = word2vec
        with open(file_name) as f:
            self._total_data = len(f.readlines()) - 1

    def __getitem__(self, index):
        line = linecache.getline(self._file_name, index + 1)
        X, y = line.strip().split()
        X = self._word2vec.get_one_hot_vector(X)
        y = self._word2vec.get_one_hot_vector(y)
        return torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return self._total_data


class Word2VecNN(nn.Module):
    def __init__(self, input_dimension, word_vec_dimensions):
        super(Word2VecNN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dimension, word_vec_dimensions),
            nn.Linear(word_vec_dimensions, input_dimension)
        )

    def forward(self, x):
        # return nn.functional.normalize(self.hidden(x), p=2, dim=-1)
        return self.hidden(x)

def skip_gram_loss(outputs, targets, word2vec: Word2VecDataset):
    losses = torch.zeros(len(outputs)).cuda()
    for i in range(len(outputs)):
        context_word_vector = targets[i]
        hidden_layer_center_word_output_vector = outputs[i]
        first_term = torch.log2(torch.nn.functional.sigmoid(torch.dot(context_word_vector, hidden_layer_center_word_output_vector)))
        k_negative_samples = word2vec.get_k_negative_samples(k=5)
        second_term = 0.0
        for neg_sample in k_negative_samples:
            second_term += torch.log2(nn.functional.sigmoid(- torch.dot(torch.Tensor(neg_sample).cuda(), hidden_layer_center_word_output_vector)))
        losses[i] = - first_term - second_term

    return torch.mean(losses)


def train(args, model, device, train_loader, optimizer, epoch, word2vec_data):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = skip_gram_loss(output, target, word2vec_data)
        writer.add_scalar("Loss/train", loss.item(), epoch)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
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
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    word2vec = Word2VecDataset('../out/sentences-small-1.txt')
    Word2VecDataset.generate_target_context_pairs(
                        window=3,
                        input_file_path='../out/sentences-small-1.txt',
                        output_file_path='../out/target-context.txt'
                    )

    dataset = SkipGramDataset('../out/target-context.txt', word2vec)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size
    )

    input_dimension = len(dataset[0][0])

    model = Word2VecNN(input_dimension, args.word_vector_dimension).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, data_loader, optimizer, epoch, word2vec)
    writer.flush()

    if args.save_model:
        torch.save(model.state_dict(), "../out/word2vec_trained.pt")


if __name__ == '__main__':
    main()
