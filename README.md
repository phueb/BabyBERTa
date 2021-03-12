<div align="center">
 <img src="images/logo.png" width="250"> 
</div>

## About

This repository contains research code for testing a small RoBERTA model trained on 
a small corpus of child-directed speech (5M words from American-English CHILDES).
Our model is implemented using the `transformers` Python package, maintained by `huggingface`.


## History

- 2020 (Spring): The BabyBERT project grew out of the BabySRL project led by Cynthia Fisher, Dan Roth, Michael Connor and Yael Gertner, 
whose published work is available [here](https://www.aclweb.org/anthology/W08-2111/). 
Having found little benefit for joint SRL and MLM training of a custom (smaller in size) version of BERT,
 a new line of research into BERT's acquisition of syntactic knowledge began. 
- 2020 (Fall): We discovered that a cognitively more plausible MLM pre-training strategy for a small BERT-like transformer outperformed an identically sized RoBERTa model, trained with standard methods in the `fairseq` library, on a large number of number agreement tasks. 
- 2021 (Spring): We are currently investigating which modifications of pre-training are most useful for acquiring syntactic knowledge in the small-model and small-data setting for Transformer language models.
 
## Probing for syntactic knowledge

Probing data can be found [here](https://github.com/phueb/Zorro). 


## BabyBERT vs. RoBERTa
 
BabyBERT is inspired by the original RoBERTa model, but departs from it in many ways.
 
Because our goal is to work with a compact model, optimized for acquiring distributional knowledge about child-directed speech,
 rather than some down-stream application, BabyBERT differs from the original BERT in the following ways:
 
- trained on American-English child-directed speech: ~5M words vs ~2B words 
- fewer hidden units and layers: ~10M parameters vs ~100M
- smaller vocabulary: ~8K vs ~30K
- masked locations are never replaced by the original or a random word
- smaller batch size: 16 vs. 256
- fewer training steps: 160K steps (approx 5-6 epochs) vs. many more in the original RoBERTa
- training examples are ordered by the age of the child to whom the utterance is directed to
- input sequences consist of 1 utterance, as opposed to multiple sentences

To achieve good performance on our number agreement tasks, 
we observed the model must not be trained with more than 1 utterance per input.
This observation hold for the CHILDES corpus, but not for other corpora.

## BabyBERT vs. fairseq RoBERTa
To train a BabyBERT like model using `fairseq`, make sure to use the following command line arguments: 

```bash
--batch-size 16
--clip-norm 1.0
--adam-betas '(0.9, 0.999)'
--weight-decay 0.0
--update-freq 1
--total-num-update 160000
--sample-break-mode eos  # one complete sentece per sample
```

There are additional subtle differences between `huggingface` and `fairseq` that prevented us from replicating our results in `fairseq`.
Potential differences include:
* weight initialisation
* floating point precision
* pre-processing (e.g. tokenization)

## Pre-processing Pipeline

1. Raw text data, in `txt` files, was previously tokenized using `spacy` which splits on contractions.
2. Sentences are separated and those that are too short or too long are excluded.
3. Multiple sentences may be combined (but default is 1) into a single sequence.
4. Each sequence is sub-word tokenized with custom-trained BBPE Tokenizer from `tokenizers`.
5. Multiple sequences are batched together (default is 16).
6. Each batch of sequences is input to a custom trained `tokenizers` BBPE Tokenizer, 
which produces output compatible with the `forward()` method of BabyBERT.


## Using the BabyBERT vocab

To use our 8192-words vocabulary for training a Roberta model in `fairseq` v0.10.2, 

```python
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig

encoder_json_path = 'data/corpora/c-n-w-8192/vocab.json'
vocab_bpe_path = 'data/corpora/c-n-w-8192/merges.txt'
cfg = GPT2BPEConfig(gpt2_encoder_json=encoder_json_path,
                    gpt2_vocab_bpe=vocab_bpe_path)
encoder = GPT2BPE(cfg)
```

The resultant object `encoder` can then be passed to `roberta.bpe` to replace the default encoder, 
 which uses 50k words.
 
To get a feeling for how this encoder splits text, use: 

```python
bpe_tokens = []
for token in encoder.bpe.re.findall(encoder.bpe.pat, text):
    token = "".join(encoder.bpe.byte_encoder[b] for b in token.encode("utf-8"))
    bpe_tokens.extend(
        bpe_token for bpe_token in encoder.bpe.bpe(token).split(" ")
    )
print(bpe_tokens)
```

## Running multiple jobs simultaneously

### Dependencies

Code in this repository is executed using [Ludwig](https://github.com/phueb/Ludwig),
 a library for running GPU-bound Python jobs on dedicated machines owned by the UIUC Learning & Language Lab.

To install all the dependencies, including `Ludwig`:

```python3
pip3 install -r requirements.txt
```
 
You will also need to obtain test sentences,
 and point `configs.Dirs.probing_sentences` to the folder where you saved them on your machine.

### Pre-training from scratch

To run 10 replications of each configuration on your machine,
 edit the path to `Zorro` probing sentences in `configs.Dirs.probing_sentences`. 
Then, type the following into the terminal:

`ludwig -r10 -i`

## Compatibility

Tested on Ubuntu 18.04, Python 3.7
