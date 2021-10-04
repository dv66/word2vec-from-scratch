# word2vec-from-scratch
Word2VecDataset model implementation from scratch.

## Dependencies Installation
* Install all dependecies.
 ```bash
pip install -r requirements.txt
  ```
* Install Pytorch 
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Training

```bash
cd src/
python model.py
usage: model.py [-h] --word-vector-dimension WORD_VECTOR_DIMENSION [--batch-size N] [--test-batch-size N] [--epochs N] --lr LR --gamma
                GAMMA [--no-cuda] [--dry-run] [--seed S] [--log-interval N] --model-name MODEL_NAME [--window-size WINDOW_SIZE]
                --n-neg-samples N_NEG_SAMPLES --corpus-file CORPUS_FILE

Word2VecDataset Implementation from scratch

optional arguments:
  -h, --help            show this help message and exit
  --word-vector-dimension WORD_VECTOR_DIMENSION
                        dimension of word embedding
  --batch-size N        input batch size for training (default: 64)
  --test-batch-size N   input batch size for testing (default: 1000)
  --epochs N            number of epochs to train (default: 14)
  --lr LR               learning rate (default: 1.0)
  --gamma GAMMA         Learning rate step gamma (default: 0.7)
  --no-cuda             disables CUDA training
  --dry-run             quickly check a single pass
  --seed S              random seed (default: 1)
  --log-interval N      how many batches to wait before logging training status
  --model-name MODEL_NAME
                        Filename for saving the current Model
  --window-size WINDOW_SIZE
                        Window size for creating target context pairs
  --n-neg-samples N_NEG_SAMPLES
                        No. of negative samples against per correct pair
  --corpus-file CORPUS_FILE
                        Text corpus file
```
## Show K Similar Words
```bash

```
