import yaml
import numpy as np
from typing import List
from pathlib import Path
from collections import OrderedDict

from babybert import configs


def save_open_ended_predictions(sentences_in: List[List[str]],
                                sentences_out: List[List[str]],
                                out_path: Path,
                                verbose: bool = False,
                                ) -> None:
    print(f'Saving open_ended probing results to {out_path}')
    with out_path.open('w') as f:
        for si, so in zip(sentences_in, sentences_out):
            for wi, wo in zip(si, so):
                line = f'{wi:>20} {wi if wi != configs.Data.mask_symbol else wo:>20}'
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


def load_utterances_from_file(file_path: Path,
                              training_order: str = 'none',
                              include_punctuation: bool = True,
                              verbose: bool = False,
                              allow_discard: bool = False,
                              ) -> List[List[str]]:
    """
    load utterances for language modeling from text file
    """

    print(f'Loading {file_path}')

    # when lower-casing, do not lower-case upper-cased symbols
    upper_cased = set(configs.Data.universal_symbols)

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
                if len(utterance) < configs.Data.min_utterance_length and allow_discard:
                    num_too_small += 1
                    continue
                if len(utterance) > configs.Data.max_utterance_length and allow_discard:
                    num_too_large += 1
                    continue

                # lower-case
                if configs.Data.lowercase_input:
                    utterance = [w if w in upper_cased else w.lower()
                                 for w in utterance]

                if not include_punctuation:
                    utterance = [w for w in utterance if w not in punctuation]

                # prevent tokenization of long words into lots of word pieces
                if configs.Data.max_word_length is not None:
                    utterance = [w if len(w) < configs.Data.max_word_length else configs.Data.long_symbol
                                 for w in utterance]
                res.append(utterance)

    if num_too_small or num_too_large:
        print(f'WARNING: Skipped {num_too_small} utterances which are shorter than {configs.Data.min_utterance_length}.')
        print(f'WARNING: Skipped {num_too_large} utterances which are larger than {configs.Data.max_utterance_length}.')

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