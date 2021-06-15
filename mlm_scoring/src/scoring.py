"""
Adapted from https://github.com/awslabs/mlm-scoring, by ph (Philip Huebner), in June, 2021.

Removed all code not directly related to mlm-scoring on BLiMP,
and added code to support huggingface Roberta model, and huggingface tokenizers.

Note:
    this script is called with working directory set to BabyBERTa/blimp.
"""

from contextlib import contextmanager
import logging
import sys
import numpy as np
from typing import List
import random
import os
import mxnet as mx
from pathlib import Path

from src.loaders import Corpus, ScoredCorpus
from src.scorers import MLMScorerPT


@contextmanager
def _stdout_to_stderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


SEED = 0


def setup_ctxs(gpu_str: str) -> List[mx.Context]:

    random.seed(SEED)
    np.random.seed(SEED)
    mx.random.seed(SEED)

    ids = [int(id) for id in gpu_str.split(',')]
    if len(ids) == 1 and ids[0] < 0:
        ctxs = [mx.cpu(0)]
    else:
        for id in ids:
            mx.random.seed(SEED, mx.gpu(id))
        ctxs = [mx.gpu(id) for id in ids]

    # Following GluonNLP's scripts/language_model/large_word_language_model.py
    # https://mxnet.incubator.apache.org/faq/env_var.html
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    os.environ['MXNET_CPU_PARALLEL_RAND_COPY'] = str(len(ctxs))
    os.environ['MXNET_CPU_WORKER_NTHREADS'] = str(len(ctxs))

    return ctxs


def score_model_on_paradigm(model,
                            vocab,
                            tokenizer,
                            path_paradigm: Path,
                            path_out_file: Path,
                            lower_case: bool,
                            ) -> None:

    ctxs = setup_ctxs(gpu_str='0')

    # Set scorer
    scorer = MLMScorerPT(model, vocab, tokenizer,
                         eos=False, wwm=False, ctxs=ctxs)

    # What data do we use?
    infile = path_paradigm.open('r')
    corpus = Corpus.from_file(infile, lower_case=lower_case)
    logging.warning("# sentences: {}".format(len(corpus)))

    # === START SHARED COMPUTATION ===

    # A scorer takes a corpus and produces a list of scores in order of the corpus
    corpus_for_scoring = corpus
    scores, true_tok_lens = scorer.score(corpus_for_scoring, ratio=1, split_size=500, per_token=False)
    scored_corpus = ScoredCorpus.from_corpus_and_scores(corpus, scores)

    num_words_list, max_sent_len = corpus.get_num_words()

    num_words_total = sum(num_words_list)
    logging.warning("# words: {}".format(num_words_total))
    logging.warning("longest sentence: {}".format(max_sent_len))

    num_toks_total = sum(true_tok_lens)
    logging.warning("# tokens: {}".format(num_toks_total))

    # === END SHARED COMPUTATION ===

    # save pseudo-log-likelihoods to disk
    scored_corpus.to_file(path_out_file, scores_only=True)
