<div align="center">
 <img src="images/logo.png" width="250"> 
</div>

## About

This repository contains research code for testing a small RoBERTA model trained on 
a small corpus of child-directed speech (5M words from American-English CHILDES).
Our model is implemented using the `transformers` Python package, maintained by `huggingface`.

## Usage

To use BabyBERTa pre-trained on AO-CHILDES, 
follow the instructions on the `huggingface` model hub, [here](https://huggingface.co/phueb/BabyBERTa-1/tree/main)

Alternatively, download this repository, install the dependencies, and then:

```python
from transformers.models.roberta import RobertaForMaskedLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('saved_models/BabyBERTa_AO-CHILDES',
                                          add_prefix_space=True,  # this must be added to produce intended behavior
                                          )
model = RobertaForMaskedLM.from_pretrained('saved_models/BabyBERTa_AO-CHILDES')
``` 

Note: Each saved checkpoint corresponds to the top-scoring model out of 10 differently initialized models on the [Zorro](https://github.com/phueb/Zorro) test suite. 

## History

- 2020 (Spring): The BabyBERTa project grew out of the BabySRL project led by Cynthia Fisher, Dan Roth, Michael Connor and Yael Gertner, 
whose published work is available [here](https://www.aclweb.org/anthology/W08-2111/). 
Having found little benefit for joint SRL and MLM training of a custom (smaller in size) version of BERT,
 a new line of research into BERT's acquisition of syntactic knowledge began. 
- 2020 (Fall): We discovered that a cognitively more plausible MLM pre-training strategy for a small BERT-like transformer outperformed an identically sized RoBERTa model, trained with standard methods in the `fairseq` library, on a large number of number agreement tasks. 
- 2021 (Spring): We investigated which modifications of pre-training are most useful for acquiring syntactic knowledge in the small-model and small-data setting for masked language models.
 
## Probing for syntactic knowledge

In order to probe BabyBERTa's grammatical knowledge, 
a test suite made of words commonly found in child-directed input is recommended. 
We developed such a test suite, [Zorro](https://github.com/phueb/Zorro). 


## BabyBERTa vs. RoBERTa
 
BabyBERTa is inspired by the original RoBERTa model, but departs from it in many ways.
 
Because our goal is to work with a compact model, optimized for acquiring distributional knowledge about child-directed speech,
 rather than some down-stream application, BabyBERTa differs from the original RoBERTa in the following ways:
 
- trained on American-English child-directed speech: ~5M words vs ~30B words 
- fewer hidden units, attention heads, and layers: ~10M parameters vs ~100M
- smaller vocabulary: ~8K vs ~50K
- masked tokens are never unmasked
- smaller batch size: 16 vs. 8K
- sentences with > 128 tokens are excluded (and never truncated)
- fewer epochs: 10 vs. approx. 40
- input sequences consist of 1 sentence, as opposed to multiple sentences

To achieve good performance on our number agreement tasks, 
we observed the model must not be trained with more than 1 sentence per input.
This observation holds for the CHILDES corpus, but not for other corpora.

## BabyBERTa vs. fairseq RoBERTa
To train a BabyBERTa like model using `fairseq`, make sure to use the following command line arguments: 

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

1. Raw text data is used as the starting point.
2. Sentences are separated and those that are too short or too long are excluded.
3. Multiple sentences may be combined (but default is 1) into a single sequence.
4. Each sequence is sub-word tokenized with custom-trained BBPE Tokenizer from `tokenizers`.
5. Multiple sequences are batched together (default is 16).
6. Each batch of sequences is input to a custom trained `tokenizers` Byte-Level BPE Tokenizer, 
which produces output compatible with the `forward()` method of BabyBERTa.


## Replicating our results

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

### A note on using Ludwig

If you want to use `ludwig` to submit training jobs, you have to understand how it interacts with `params.py`. For example, if you set,

```
param2requests = {
    'training_order': ['original', 'reversed'], 
    'consecutive_masking': [True],
}
```

and then run `ludwig -i`, you will train BabyBERTa once on the unmodified training corpus, and again on the same corpus in reverse order. Because `'consecutive_masking'` only has a single value provided to it in the list, this will not result in additional models being trained. Rather, both models will be trained with `'consecutive_masking'` set to `True`. 
The same is true of any parameter in param2requests. So,

```
param2requests = {
    'batch_size': [16, 32], 
    'consecutive_masking': [True, False],

}
```

will result in 2x2=4 models being trained.
The following,

```
param2requests = {
    'training_order': ['original', 'reversed']
    'batch_size': [16, 32, 64], 
    'consecutive_masking': [True, False],

}
```

will result in 2x3x2=12 models being trained. 
You can ignore `param2debug`. It works exactly the same way as `param2requests`, except that it will be used instead of `param2requests` when you use the `-d` flag. 

Note that each job both trains and evaluates a model at differenct checkpoints. The results of evaluation are saved to a folder that will created on your system. Each hyper parameter configuration has a dedicated folder. If you run multiple replications for each configuration (e.g. using the flag `-r`), each folder will be populated with additional sub-folders, one for each replication. 

## Compatibility

Tested on Ubuntu 18.04, Python 3.8
