import yaml
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

from babyberta import configs


def save_open_ended_predictions(raw_sentences: List[str],
                                predicted_words: List[str],
                                out_path: Path,
                                verbose: bool = False,
                                ) -> None:
    print(f'Saving open_ended probing results to {out_path}')
    with out_path.open('w') as f:
        for rs, pw in zip(raw_sentences, predicted_words):
            for rw in rs.split():
                line = f'{rw:>20} {rw if rw != configs.Data.mask_symbol else pw:>20}'
                f.write(line + '\n')
                if verbose:
                    print(line)
            f.write('\n')
            if verbose:
                print('\n')


def save_forced_choice_predictions(raw_sentences: List[str],
                                   cross_entropies: List[float],
                                   out_path: Path,
                                   verbose: bool = False,
                                   ) -> None:
    print(f'Saving forced_choice probing results to {out_path}')
    with out_path.open('w') as f:
        for s, xe in zip(raw_sentences, cross_entropies):
            line = f'{s} {xe:.4f}'
            f.write(line + '\n')
            if verbose:
                print(line)


def load_sentences_from_file(file_path: Path,
                             training_order: str = 'none',
                             include_punctuation: bool = True,
                             verbose: bool = False,
                             allow_discard: bool = False,
                             ) -> List[str]:
    """
    load sentences for language modeling from text file
    """

    print(f'Loading {file_path}', flush=True)

    # when lower-casing, do not lower-case upper-cased symbols
    upper_cased = set(configs.Data.roberta_symbols)

    res = []
    num_too_small = 0
    with file_path.open('r') as line_by_line_file:

        for sentence in line_by_line_file.readlines():

            if not sentence:  # during probing, parsing logic above may produce empty sentences
                continue

            sentence = sentence.rstrip('\n')

            # check  length
            if sentence.count(' ') < configs.Data.min_sentence_length - 1 and allow_discard:
                num_too_small += 1
                continue

            if not include_punctuation:
                sentence = sentence.rstrip('.')
                sentence = sentence.rstrip('!')
                sentence = sentence.rstrip('?')

            res.append(sentence)

    if num_too_small:
        print(f'WARNING: Skipped {num_too_small:,} sentences which are shorter than {configs.Data.min_sentence_length}.')

    if training_order in ['none', 'age-ordered']:
        pass
    elif training_order == 'age-reversed':
        res = res[::-1]
    else:
        raise AttributeError('Invalid arg to "training_order".')

    return res


def save_yaml_file(path_out: Path,
                   param2val: Dict[str, Any],
                   ):
    if not path_out.parent.exists():
        path_out.parent.mkdir()
    with path_out.open('w', encoding='utf8') as f:
        yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)