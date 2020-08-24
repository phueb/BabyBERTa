<div align="center">
 <img src="images/logo.png" width="250"> 
</div>

## Background

This repository contains research code that compares syntactic abilities of BERT trained on 
a small cognitively plausible corpus of child-directed speech (5M words from American-English CHILDES) 
to that trained on a large (standard) adult-generated text.

The code is for research purpose only. 
The goal of this research project is to understand language acquisition from the point of view of distributional learning models.

## History

- 2020 (Spring): The BabyBERT project grew out of the BabySRL project led by Cynthia Fisher, Dan Roth, Michael Connor and Yael Gertner, 
whose published work is available [here](https://www.aclweb.org/anthology/W08-2111/). 
Having found little benefit for joint SRL and MLM training of a custom (smaller in size) version of BERT,
 a new line of research into BERT's success on syntactic task was begun. 
 
## Probing for syntactic knowledge

Probing data can be found [here](https://github.com/phueb/Babeval). 


## BabyBERT vs. BERT
 
BabyBERT is inspired by the original BERT model, but departs from it in many ways.
 
Because our goal is to work with a compact model, optimized for acquiring distributional knowledge about child-directed speech,
 rather than some down-stream application, BabyBERT differs from the original BERT in the following ways:
 
0. trained on American-English child-directed speech: ~5M words vs ~2B words 
1. fewer hidden units and layers: ~10M parameters vs ~100M
2. smaller vocabulary: ~8K vs ~30K
3. no next-sentence prediction objective (as in RoBERTa)
4. dynamic masking: the same word is never masked more than once in the same utterance (as in RoBERTa)
5. only 1 word per utterance is masked, and masked locations are never replaced by the original or a random word
6. no learning rate schedule
7. smaller batch size (16), and larger learning rate (1e-4)
8. only 1 complete pass through training data: 1 epoch vs. ~30 epochs
9. training examples are ordered by the age of the child to whom the utterance is directed to

Contrary to the original BERT implementation, no next-sentence prediction objective is used during training, 
as was done in the original implementation. 
This reduces training time, code complexity and [learning two separate semantic spaces](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1117&context=scil).


## Using the code

Code in this repository is executed using `Ludwig`,
 a library for running GPU-bound Python jobs on dedicated machines owned by the UIUC Learning & Language Lab.
 
You will also need to obtain test sentences,
 and point `configs.Dirs.probing_sentneces` to the folder where you saved them on your machine.

### Pre-training from scratch

To run 10 replications of each configuration on your machine, type the following into the terminal:

`ludwig -r10 -i`

### Probing pre-trained models

Run the script `scripts/probe_pretrained_models`

## Compatibility

Tested on Ubuntu 16.04, Python 3.7, transformers=3.02, and torch==1.2.0
