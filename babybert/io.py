import yaml
import numpy as np
from typing import List
from pathlib import Path

from babybert import configs


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
    upper_cased = set(configs.Data.universal_symbols)

    tokenized_sentences = []
    punctuation = {'.', '?', '!'}
    num_too_small = 0
    with file_path.open('r') as f:

        for line in f.readlines():

            # tokenize transcript
            transcript = line.strip().split()  # a transcript containing multiple sentences
            transcript = [w for w in transcript]

            # split transcript into sentences
            tokenized_sentences_in_transcript = [[]]
            for w in transcript:
                tokenized_sentences_in_transcript[-1].append(w)
                if w in punctuation:
                    tokenized_sentences_in_transcript.append([])

            # collect sentences
            for ts in tokenized_sentences_in_transcript:

                if not ts:  # during probing, parsing logic above may produce empty sentences
                    continue

                # check  length
                if len(ts) < configs.Data.min_sentence_length and allow_discard:
                    num_too_small += 1
                    continue

                # lower-case
                ts = [w if w in upper_cased else w.lower()
                      for w in ts]

                if not include_punctuation:
                    ts = [w for w in ts if w not in punctuation]

                # prevent tokenization of long words into lots of sub-tokens
                if configs.Data.max_word_length is not None:
                    ts = [w if len(w) <= configs.Data.max_word_length else configs.Data.long_symbol
                          for w in ts]
                tokenized_sentences.append(ts)

    if num_too_small:
        print(f'WARNING: Skipped {num_too_small:,} sentences which are shorter than {configs.Data.min_sentence_length}.')

    if verbose:
        lengths = [len(u) for u in tokenized_sentences]
        print('Found {:,} sentences'.format(len(tokenized_sentences)))
        print(f'Min    sentence length: {np.min(lengths):.2f}')
        print(f'Max    sentence length: {np.max(lengths):.2f}')
        print(f'Mean   sentence length: {np.mean(lengths):.2f}')
        print(f'Median sentence length: {np.median(lengths):.2f}')

    if training_order in ['none', 'age-ordered']:
        pass
    elif training_order == 'age-reversed':
        tokenized_sentences = tokenized_sentences[::-1]
    else:
        raise AttributeError('Invalid arg to "training_order".')

    sentences = [' '.join(ts) for ts in tokenized_sentences]
    return sentences


def save_yaml_file(param2val_path: Path,
                   architecture: str,
                   ):
    param2val = {'architecture': architecture}
    with param2val_path.open('w', encoding='utf8') as f:
        yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)