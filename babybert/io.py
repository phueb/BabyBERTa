import yaml
import numpy as np
from typing import List
from pathlib import Path
from collections import OrderedDict

from babybert import configs


def save_open_ended_predictions(sentences_in: List[List[str]],
                                predicted_words: List[List[str]],
                                out_path: Path,
                                verbose: bool = False,
                                ) -> None:
    print(f'Saving open_ended probing results to {out_path}')
    with out_path.open('w') as f:
        for s, pw in zip(sentences_in, predicted_words):
            for w in s:
                line = f'{w:>20} {w if w != "[MASK]" else pw:>20}'
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


def make_vocab(childes_vocab_file: Path,
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
        to_index += sorted(set(childes_vocab + google_vocab_cleaned))
    elif google_vocab_rule == 'exclusive':
        to_index += google_vocab_cleaned
    elif google_vocab_rule == 'none':
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
    assert index == len(res), (index, len(res))

    # the transformers.PreTrainedTokenizer doesn't respect the order below, but it doesn't matter.
    # it's good to check this order here, so that the special tokens show up in first few lines in vocab file
    assert res['[PAD]'] == 0
    assert res['[UNK]'] == 1
    assert res['[CLS]'] == 2
    assert res['[SEP]'] == 3
    assert res['[MASK]'] == 4

    return res


def load_utterances_from_file(file_path: Path,
                              training_order: str = 'none',
                              verbose: bool = True,
                              allow_discard: bool = False) -> List[List[str]]:
    """
    load utterances for language modeling from text file
    """

    print(f'Loading {file_path}')

    # when lower-casing, do not lower-case upper-cased symbols
    upper_cased = set(configs.Data.special_symbols + configs.Data.childes_symbols)

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

    if num_too_small or num_too_large:
        print(f'WARNING: Skipped {num_too_small} utterances which are shorter than {configs.Data.min_seq_length}.')
        print(f'WARNING: Skipped {num_too_large} utterances which are larger than {configs.Data.max_seq_length}.')

    if verbose:
        lengths = [len(u) for u in res]
        print('Found {:,} utterances'.format(len(res)))
        print(f'Min    utterance length: {np.min(lengths):.2f}')
        print(f'Max    utterance length: {np.max(lengths):.2f}')
        print(f'Mean   utterance length: {np.mean(lengths):.2f}')
        print(f'Median utterance length: {np.median(lengths):.2f}')

    if training_order in ['none', 'age-ordered']:
        pass
    elif training_order == 'age-reversed':
        res = res[::-1]
    else:
        raise AttributeError('Invalid arg to "training_order".')

    return res


def save_yaml_file(param2val_path: Path,
                   architecture: str,
                   ):
    param2val = {'architecture': architecture}
    with param2val_path.open('w', encoding='utf8') as f:
        yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)