import yaml
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path

from tokenizers import Tokenizer

from babyberta import configs


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
                             include_punctuation: bool = True,
                             allow_discard: bool = False,
                             ) -> List[str]:
    """
    load sentences for language modeling from text file
    """

    print(f'Loading {file_path}', flush=True)

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

    return res


def save_yaml_file(path_out: Path,
                   param2val: Dict[str, Any],
                   ):
    if not path_out.parent.exists():
        path_out.parent.mkdir()
    with path_out.open('w', encoding='utf8') as f:
        yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)


def load_tokenizer(config_path: Path,
                   max_input_length: int,
                   ) -> Tokenizer:

    tokenizer = Tokenizer.from_file(str(config_path))
    tokenizer.enable_truncation(max_length=max_input_length)

    return tokenizer


def load_wikipedia_sentences(input_filepath: Path,
                             percent: int,
                             shift: int,
                             ) -> List[str]:
    """
    return a sample of wiki sentences from a large text file, built using witokit.

    """

    if not 0 < percent < 100:
        raise Exception('Specified percent param should be in ]0, 100[')
    print('Sampling input file {}'.format(input_filepath))

    print('Counting number of lines in file...')
    with input_filepath.open('r', encoding='utf-8') as input_stream:
        num_lines = sum(1 for x in input_stream)
    print(f'Number of lines in {input_filepath}={num_lines:,}')
    final_count = num_lines * percent / 100
    sampling = num_lines / final_count

    # collect sentences
    res = []
    with open(input_filepath, 'r', encoding='utf-8') as input_stream:
        for idx, line in enumerate(input_stream):
            if (idx + shift) % round(sampling) == 0:
                res.append(line.strip())

    return res