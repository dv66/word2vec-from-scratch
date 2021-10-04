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
$ python k_similar_words.py --model-file=../frozen-models/word2vec_trained-original_corpus-cleaned-20k.en-3.pt \
--vocabulary-pickle-file=../frozen-models/original_corpus-cleaned-20k.en.pkl \
--reference-word SCARED \
--k=20
[(1.0, 18098), (0.9963043928146362, 11017), (0.9961966276168823, 16533), (0.9961528182029724, 5540), (0.9960448741912842, 10079), (0.9960080981254578, 7070), (0.995915412902832, 12760), (0.9959071278572083, 5178), (0.9958705306053162, 21026), (0.9958584308624268, 706), (0.9958078265190125, 14596), (0.9957879781723022, 5894), (0.9957615733146667, 815), (0.9957253336906433, 9667), (0.9957132935523987, 20932), (0.9956495761871338, 11005), (0.9956390261650085, 9464), (0.9956069588661194, 20101), (0.9955527782440186, 735), (0.9955347180366516, 19486)]
['SCARED', 'FATIMA', 'RAPTURES', 'BADMASH', 'OUS', 'IRRITABLE', 'TRICKS', 'ARRESTING', 'INSPIRE', 'EMSWORTH', 'PLEADED', 'RESTLESS', 'OFFEND', 'CONTRADICT', 'EXPRESSLY', 'SHORT:', 'TELEGRAPHED', 'SKEPTIC', 'SUTRO', 'JOYS']



$ python k_similar_words.py --model-file=../frozen-models/word2vec_trained-original_corpus-cleaned-20k.en-1.pt \
--vocabulary-pickle-file=../frozen-models/original_corpus-cleaned-20k.en.pkl \
--reference-word GOVERNMENT \
--k=20
[(1.0, 9016), (0.992904782295227, 1279), (0.992862343788147, 16564), (0.9927070140838623, 13065), (0.9926716089248657, 6404), (0.9926643371582031, 19109), (0.9926505088806152, 12541), (0.9925747513771057, 15803), (0.9925470948219299, 2684), (0.9925264120101929, 13641), (0.9925150871276855, 8274), (0.9924643635749817, 6259), (0.9924466013908386, 9061), (0.9924026727676392, 6374), (0.9923996925354004, 10714), (0.9923939108848572, 19323), (0.9923858046531677, 6965), (0.9923484921455383, 5984), (0.9923253059387207, 12290), (0.9923092126846313, 144)]
['GOVERNMENT', 'TECHNOLOGY', 'NATURES', 'RIGHTS', 'NATIONS', 'SHOWS', 'RESPONSIBILITY', 'SACRIFICE', 'PHYSICAL', 'TERRITORY', 'RELIGION', 'RELIGIONS', 'CULTURAL', 'FUNDAMENTAL', 'INDIVIDUAL', 'MYTHS', 'RULED', 'EXISTENCE', 'SUPERIOR', 'STYLE']


$ python k_similar_words.py --model-file=../frozen-models/word2vec_trained-original_corpus-cleaned-20k.en-3.pt \
--vocabulary-pickle-file=../frozen-models/original_corpus-cleaned-20k.en.pkl \
--reference-word FIFTEEN \
--k=20
[(1.0, 20346), (0.9680690765380859, 16769), (0.9678809642791748, 8499), (0.9675710797309875, 7044), (0.9656014442443848, 14075), (0.9637587666511536, 3649), (0.9631183743476868, 7080), (0.9599475860595703, 18503), (0.9588974714279175, 11972), (0.9574429392814636, 4057), (0.9573668241500854, 3165), (0.9564999938011169, 12354), (0.9563896059989929, 7761), (0.9563530087471008, 13768), (0.9556819200515747, 2833), (0.9555959105491638, 15549), (0.9553748965263367, 1613), (0.9552883505821228, 12757), (0.9548815488815308, 20985), (0.9548521041870117, 14470)]
['FIFTEEN', 'TWELVE', 'ELEVEN', 'NINE', 'OCLOCK', 'SIXTY', 'WEEKS', 'SEVENTY', 'HOURS', 'NIGHTS', 'SECONDS', 'CARBUNCLES', 'SPENT', 'CONFERENCES', 'NINETEEN', 'POPPED', 'SIXTEEN', 'YARDS', 'WEEK', 'STABILISING']


```
