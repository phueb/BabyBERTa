from typing import List, Tuple, Generator, Union, Optional
import random
from itertools import combinations
import pyprind
import torch
from transformers import RobertaTokenizerFast

from babybert.masking import tokenize_and_mask
from babybert.utils import RobertaInput
from babybert import configs


class Batcher:

    def __init__(self,
                 sequences: List[str],
                 tokenizer: RobertaTokenizerFast,
                 batch_size: int,
                 num_mask_patterns: int,
                 mask_pattern_size: int,
                 allow_truncated_sentences: bool,
                 ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_mask_patterns = num_mask_patterns
        self.mask_pattern_size = mask_pattern_size
        self.allow_truncated_sentences = allow_truncated_sentences

    def _gen_make_mask_patterns(self,
                                num_tokens: int,
                                ) -> Generator[Tuple[str, Tuple[int]], None, None]:
        """
        make all mask patterns for given sequence.

        a mask_pattern is a tuple of 1 or more integers
        corresponding to the indices of a tokenized sequence that should be masked.

        notes:
        - pattern size is dynamically shortened if a tokenized sequence is smaller than mask_pattern_size.
        - num_mask_patterns is dynamically adjusted if number of possible patterns is smaller than num_mask_patterns.
        """

        num_tokens = min(configs.Data.max_num_tokens_in_sequence - 2,  # -2 because we need to fit eos and bos symbols
                         num_tokens)  # prevent masking of token in overflow region
        pattern_size = min(self.mask_pattern_size, num_tokens)

        # sample patterns from population of all possible patterns
        all_mask_patterns = list(combinations(range(num_tokens), pattern_size))
        num_patterns = min(self.num_mask_patterns, len(all_mask_patterns))
        for mask_pattern in random.sample(all_mask_patterns, k=num_patterns):
            yield mask_pattern

    def _add_mask_patterns(self) -> Generator[Tuple[str, Tuple[int]], None, None]:
        pbar = pyprind.ProgBar(len(self.sequences))
        num_too_large = 0
        for s in self.sequences:
            tokens = self.tokenizer.tokenize(s, add_special_tokens=False)
            num_tokens = len(tokens)
            num_tokens_and_special_symbols = num_tokens + 2

            # exclude sequence if too many tokens
            if not self.allow_truncated_sentences and num_tokens_and_special_symbols > configs.Data.max_num_tokens_in_sequence:
                num_too_large += 1
                continue

            for mp in self._gen_make_mask_patterns(num_tokens):
                yield s, mp
            pbar.update()

        print()
        print(f'Excluded {num_too_large} sequences with more than {configs.Data.max_num_tokens_in_sequence} tokens.',
              flush=True)

    def gen_batch_sized_chunks(self,
                               consecutive_masking: bool,
                               ) -> Generator[Tuple[List[str], List[Tuple[int]]], None, None]:
        """
        create batches in one of two ways:
        1) consecutive=true: sequences differing only in mask pattern are put in same batch.
        2) consecutive=false: sequences differing only in mask pattern are not put in same batch.
        """
        print('Adding mask patterns...', flush=True)
        ss_mps = list(self._add_mask_patterns())  # list of sequences, each with a mask pattern
        if not consecutive_masking:
            random.shuffle(ss_mps)
        print('Done')

        # split data into batch-sized chunks
        for start in range(0, len(ss_mps), self.batch_size):
            end = min(len(ss_mps), start + self.batch_size)
            sequences_in_batch, mask_patterns = zip(*ss_mps[start:end])
            yield list(sequences_in_batch), list(mask_patterns)


def gen_batches(sequences: List[str],
                tokenizer: RobertaTokenizerFast,
                params,
                batch_size: Optional[int] = None,  # option to use larger-than-training batch size to speed eval
                ) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None]], None, None]:

    """
    generate batches of vectorized data ready for training.

    notes:
    -set num_maks_patterns to 1 and mask_pattern_size to 0 when probing
    """

    # create mask patterns + select which sequences are put in same batch (based on patterns)
    batcher = Batcher(sequences,
                      tokenizer,
                      batch_size or params.batch_size,
                      params.num_mask_patterns,
                      params.mask_pattern_size,
                      params.allow_truncated_sentences)
    for sequences_in_batch, mask_patterns in batcher.gen_batch_sized_chunks(params.consecutive_masking):
        yield from tokenize_and_mask(sequences_in_batch, mask_patterns, tokenizer)
