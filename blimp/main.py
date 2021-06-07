import argparse
from contextlib import contextmanager
import logging
import math
import sys
import numpy as np
from typing import List
import random
import os
import mxnet as mx

from transformers.models.roberta import RobertaForMaskedLM
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from mlm_scoring.loaders import Corpus, ScoredCorpus
from mlm_scoring.scorers import MLMScorerPT


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



def cmd_score(args: argparse.Namespace) -> None:

    # this function is called with working directory set to BabyBERTa/blimp

    ctxs = setup_ctxs(gpu_str='0')


    # ph
    model = RobertaForMaskedLM.from_pretrained(f'saved_models/{args.model}')
    tokenizer = Tokenizer.from_file('../data/tokenizers/a-a-w-w-w-8192.json')
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair=None,
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )
    pad_symbol = '<pad>'
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad_symbol), pad_token=pad_symbol)
    tokenizer.enable_truncation(max_length=128)
    vocab = tokenizer.get_vocab()

    # Set scorer
    scorer = MLMScorerPT(model, vocab, tokenizer,
                         eos=args.eos, wwm=args.whole_word_mask, capitalize=None, ctxs=ctxs)

    # What data do we use?
    corpus = Corpus.from_file(args.infile, max_utts=args.max_utts)
    logging.warning("# sentences: {}".format(len(corpus)))

    # === START SHARED COMPUTATION ===

    # A scorer takes a corpus and produces a list of scores in order of the corpus
    corpus_for_scoring = corpus
    scores, true_tok_lens = scorer.score(corpus_for_scoring, ratio=1, split_size=args.split_size,
                                         per_token=args.per_token)
    scored_corpus = ScoredCorpus.from_corpus_and_scores(corpus, scores)

    num_words_list, max_sent_len = corpus.get_num_words()
    if args.eos:
        logging.warning("Adding EOSes '.' to (P)PPL computation")
        num_words_list = [x + 1 for x in num_words_list]
    num_words_total = sum(num_words_list)
    if args.eos:
        logging.warning("# words (excluding EOS '.'): {}".format(num_words_total))
    else:
        logging.warning("# words: {}".format(num_words_total))
    logging.warning("longest sentence: {}".format(max_sent_len))

    num_toks_total = sum(true_tok_lens)
    if args.eos:
        logging.warning("# tokens (including EOS '.'): {}".format(num_toks_total))
    else:
        logging.warning("# tokens: {}".format(num_toks_total))

    if not args.per_token:
        plls = np.array(scores)
        pppl_tok_micro = np.exp(- plls.sum() / num_toks_total).item()
        logging.warning("Token-level (P)PPL: {}".format(pppl_tok_micro))

        # pppl_tok_macro = np.exp(- (plls / np.array(true_tok_lens)).mean())
        # logging.warning("Token-level (P)PPL, macro: {}".format(pppl_tok_macro))

        pppl_word_micro = math.exp((num_toks_total / num_words_total) * math.log(pppl_tok_micro))
        logging.warning("Word-normalized (P)PPL: {}".format(pppl_word_micro))

        # pppl_word_macro = np.exp(- (plls / np.array(num_words_list)).mean())
        # logging.warning("Word-normalized (P)PPL, macro: {}".format(pppl_word_macro))

    # === END SHARED COMPUTATION ===

    # How do we output?
    scored_corpus.to_file(sys.stdout, scores_only=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Masked Language Model Scoring")

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    parser.add_argument('--gpus', type=str, default='-1',
                        help="Comma-delimited list of GPUs to use (-1 is CPU)")
    parser.add_argument('--max-utts', type=int,
                        help="maximum utterances to parse")
    parser.add_argument('--model', type=str,
                        help="Model to (re)score")
    parser.add_argument('--weights', type=str, default=None,
                        help="Model weights to load")
    parser.add_argument('--mode', type=str, choices=['ref', 'hyp'],
                        help="Scoring references (.txt, .json 'refs') vs. hypotheses (.json 'hyp_*')")
    parser.add_argument('--temp', type=float, default=1.0,
                        help="softmax temperature")
    parser.add_argument('--split-size', type=int, default=500,
                        help="split size (per GPU)")
    parser.add_argument('--no-mask', action='store_true',
                        help="Instead of making masked copies, do not mask")
    parser.add_argument('--tgt', type=str, default='en',
                        help="Code to use for language embeddings, where appropriate")
    parser.add_argument('--eos', action='store_true',
                        help="append '.' (this can help mitigate train-test disparity)")
    parser.add_argument('--whole-word-mask', action='store_true',
                        help="mask whole words")
    parser.add_argument('--per-token', action='store_true',
                        help="output lists of per-token scores (slower)")
    parser.add_argument('infile', nargs='?', type=argparse.FileType('rt'),
                        help="File to score (.json = ESPNet JSON, otherwise newline-separated text). Loads whole file into memory!")

    arguments = parser.parse_args()
    cmd_score(arguments)
