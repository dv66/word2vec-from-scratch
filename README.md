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
After training, a model file (.pt) and a vocabulary pickle file (.pkl) will be generated.
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
python k_similar_words.py -h
usage: k_similar_words.py [-h] --model-file MODEL_FILE --vocabulary-pickle-file VOCABULARY_PICKLE_FILE --reference-word REFERENCE_WORD --k K

Print K Similar Words for a given word

optional arguments:
  -h, --help            show this help message and exit
  --model-file MODEL_FILE
                        Trained word2vec model file path
  --vocabulary-pickle-file VOCABULARY_PICKLE_FILE
                        Pickle file for vocabulary corresponding to trained model
  --reference-word REFERENCE_WORD
                        Reference word for similar words
  --k K                 Number of similar words to print

```
Example:
```bash
$ python k_similar_words.py --model-file=../frozen-models/word2vec_trained-original_corpus-cleaned-20k.en-1.pt  --vocabulary-pickle-file=../frozen-models/original_corpus-cleaned-20k.en.pkl --reference-word SCARED --k=10

[(1.0, 18098), (0.9966104626655579, 3463), (0.9966039061546326, 250), (0.9965393543243408, 6128), (0.9964630603790283, 14955), (0.9964352250099182, 11893), (0.9964273571968079, 17516), (0.9964168071746826, 21240), (0.9964138269424438, 6685), (0.9964002966880798, 1262)]
['SCARED', 'LONELY', 'ANYWHERE', 'VILLAGER', 'DRAWERS', 'OBSERVE', 'TENSION', 'PUBLISH', 'PROCLAIMED', 'INFORMED']
```
