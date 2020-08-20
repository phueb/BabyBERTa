import numpy as np
from typing import List
from pathlib import Path
from collections import OrderedDict

from babybert import configs


# when lower-casing, do not lower-case upper-cased symbols
upper_cased = configs.Data.special_symbols + configs.Data.childes_symbols  # order matters


def save_open_ended_predictions(sentences_in: List[List[str]],
                                sentences_out: List[List[str]],
                                out_path: Path,
                                verbose: bool = False,
                                ) -> None:
    print(f'Saving open_ended probing results to {out_path}')
    with out_path.open('w') as f:
        for s1i, s2i in zip(sentences_in, sentences_out):
            assert len(s1i) == len(s2i)
            for ai, bi, ci in zip(s1i, s2i):  # careful, zips over shortest list
                line = f'{ai:>20} {bi:>20}'
                f.write(line + '\n')
                if verbose:
                    print(line)
            f.write('\n')
            if verbose:
                print('\n')


def save_forced_choice_predictions(mlm_in,
                                   cross_entropies,
                                   out_path: Path,
                                   verbose: bool = False,
                                   ) -> None:
    print(f'Saving forced_choice probing results to {out_path}')
    with out_path.open('w') as f:
        for s, xe in zip(mlm_in, cross_entropies):
            line = f'{" ".join(s)} {xe:.4f}'
            f.write(line + '\n')
            if verbose:
                print(line)


def load_words_from_vocab_file(vocab_file: Path,
                               col: int = 0):

    res = []
    with vocab_file.open("r", encoding="utf-8") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            token = line.split()[col]

            # exclude word with non-ASCII characters
            if [True for c in token if ord(c) > 127]:
                continue

            res.append(token)
    return res


def load_vocab(childes_vocab_file: Path,
               google_vocab_file: Path,
               vocab_size: int,  # childes-vocab, not total vocab
               google_vocab_rule: str) -> OrderedDict:

    childes_vocab = load_words_from_vocab_file(childes_vocab_file, col=1)[:vocab_size]
    google_vocab = load_words_from_vocab_file(google_vocab_file, col=0)

    # exclude any google wordpieces not in CHILDES vocab, but leave non-start wordpieces (e.g. ##s)
    google_vocab_cleaned = [w for w in google_vocab
                            if w in set(childes_vocab) or w.startswith('##')]

    # init
    to_index = configs.Data.special_symbols + configs.Data.childes_symbols

    # add from childes vocab
    if google_vocab_rule == 'inclusive':
        to_index += set(childes_vocab + google_vocab_cleaned)
    elif google_vocab_rule == 'exclusive':
        to_index += google_vocab_cleaned
    elif google_vocab_rule == 'excluded':
        to_index += childes_vocab
    else:
        raise AttributeError('Invalid arg to "google_vocab_rule".')

    # index
    res = OrderedDict()
    index = 0
    for token in to_index:
        if token in res:
            # happens for symbols
            continue
        res[token] = index
        index += 1

    assert len(set(res)) == len(res)
    assert res['[PAD]'] == 0
    assert index == len(res), (index, len(res))

    return res


def load_utterances_from_file(file_path: Path,
                              verbose: bool = False,
                              allow_discard: bool = False) -> List[List[str]]:
    """
    load utterances for language modeling from text file
    """

    print(f'Loading {file_path}')

    res = []
    punctuation = {'.', '?', '!'}
    num_too_small = 0
    num_too_large = 0
    with file_path.open('r') as f:

        for line in f.readlines():

            # tokenize transcript
            transcript = line.strip().split()  # a transcript containing multiple utterances
            transcript = [w for w in transcript]

            # split transcript into utterances
            utterances = [[]]
            for w in transcript:
                utterances[-1].append(w)
                if w in punctuation:
                    utterances.append([])

            # collect utterances
            for utterance in utterances:

                if not utterance:  # during probing, parsing logic above may produce empty utterances
                    continue

                # check  length
                if len(utterance) < configs.Data.min_seq_length and allow_discard:
                    num_too_small += 1
                    continue
                if len(utterance) > configs.Data.max_seq_length and allow_discard:
                    num_too_large += 1
                    continue

                # lower-case
                if configs.Data.uncased:
                    utterance = [w if w in upper_cased else w.lower()
                                 for w in utterance]

                res.append(utterance)

    print(f'WARNING: Skipped {num_too_small} utterances which are shorter than {configs.Data.min_seq_length}.')
    print(f'WARNING: Skipped {num_too_large} utterances which are larger than {configs.Data.max_seq_length}.')

    if verbose:
        lengths = [len(u) for u in res]
        print('Found {:,} utterances'.format(len(res)))
        print(f'Max    utterance length: {np.max(lengths):.2f}')
        print(f'Mean   utterance length: {np.mean(lengths):.2f}')
        print(f'Median utterance length: {np.median(lengths):.2f}')

    return res
