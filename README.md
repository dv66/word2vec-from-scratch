# word2vec-from-scratch
Word2VecDataset model implementation from scratch.

## Training

### Setup
* Install all dependecies.
 ```bash
pip install -r requirements.txt
  ```
* Install Pytorch 
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Usage

```bash
$ cd src/
$ python pipeline.py -h
usage: pipeline.py [-h] --input_dir PATH --output_dir PATH --src_lang SRC_LANG
                   --tgt_lang TGT_LANG
                   [--validation_samples VALIDATION_SAMPLES]
                   [--src_seq_length SRC_SEQ_LENGTH]
                   [--tgt_seq_length TGT_SEQ_LENGTH]
                   [--model_prefix MODEL_PREFIX] [--eval_model PATH]
                   [--train_steps TRAIN_STEPS]
                   [--train_batch_size TRAIN_BATCH_SIZE]
                   [--eval_batch_size EVAL_BATCH_SIZE]
                   [--gradient_accum GRADIENT_ACCUM]
                   [--warmup_steps WARMUP_STEPS]
                   [--learning_rate LEARNING_RATE] [--layers LAYERS]
                   [--rnn_size RNN_SIZE] [--word_vec_size WORD_VEC_SIZE]
                   [--transformer_ff TRANSFORMER_FF] [--heads HEADS]
                   [--valid_steps VALID_STEPS]
                   [--save_checkpoint_steps SAVE_CHECKPOINT_STEPS]
                   [--average_last AVERAGE_LAST] [--world_size WORLD_SIZE]
                   [--gpu_ranks [GPU_RANKS [GPU_RANKS ...]]]
                   [--train_from TRAIN_FROM] [--do_train] [--do_eval]
                   [--do_preprocess] [--nbest NBEST] [--alpha ALPHA]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir PATH, -i PATH
                        Input directory
  --output_dir PATH, -o PATH
                        Output directory
  --src_lang SRC_LANG   Source language
  --tgt_lang TGT_LANG   Target language
  --validation_samples VALIDATION_SAMPLES
                        no. of validation samples to take out from train
                        dataset when no validation data is present
  --src_seq_length SRC_SEQ_LENGTH
                        maximum source sequence length
  --tgt_seq_length TGT_SEQ_LENGTH
                        maximum target sequence length
  --model_prefix MODEL_PREFIX
                        Prefix of the model to save
  --eval_model PATH     Path to the specific model to evaluate
  --train_steps TRAIN_STEPS
                        no of training steps
  --train_batch_size TRAIN_BATCH_SIZE
                        training batch size (in tokens)
  --eval_batch_size EVAL_BATCH_SIZE
                        evaluation batch size (in sentences)
  --gradient_accum GRADIENT_ACCUM
                        gradient accum
  --warmup_steps WARMUP_STEPS
                        warmup steps
  --learning_rate LEARNING_RATE
                        learning rate
  --layers LAYERS       layers
  --rnn_size RNN_SIZE   rnn size
  --word_vec_size WORD_VEC_SIZE
                        word vector size
  --transformer_ff TRANSFORMER_FF
                        transformer feed forward size
  --heads HEADS         no of heads
  --valid_steps VALID_STEPS
                        validation interval
  --save_checkpoint_steps SAVE_CHECKPOINT_STEPS
                        model saving interval
  --average_last AVERAGE_LAST
                        average last X models
  --world_size WORLD_SIZE
                        world size
  --gpu_ranks [GPU_RANKS [GPU_RANKS ...]]
                        gpu ranks
  --train_from TRAIN_FROM
                        start training from this checkpoint
  --do_train            Run training
  --do_eval             Run evaluation
  --do_preprocess       Run the preprocessor to create binary files. Delete
                        previous files if theres any.
  --nbest NBEST         sentencepiece nbest size
  --alpha ALPHA         sentencepiece alpha
```

