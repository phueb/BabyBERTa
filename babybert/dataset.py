from typing import List, Tuple, Generator, Union, Optional
import random
from itertools import combinations


import numpy as np
import pyprind
import torch
from transformers import RobertaTokenizerFast
from transformers.modeling_roberta import create_position_ids_from_input_ids


from babybert import configs
from babybert.utils import RobertaInput
from babybert.params import Params


class ProbingParams:
    sample_with_replacement = False
    max_num_tokens_in_sequence = 256
    leave_unmasked_prob = 0.0
    random_token_prob = 0.0
    consecutive_masking = None
    mask_pattern_size = None
    num_mask_patterns = None
    allow_truncated_sentences = False
    batch_size = 32


class DataSet:

    @classmethod
    def for_probing(cls,
                    sequences: List[str],
                    tokenizer: RobertaTokenizerFast,
                    ):
        """
        returns instance when used for probing.
        different in that mask patterns are determined from input
        """

        def _get_mask_pattern_from_probing_sequence(sequence: str,
                                                    ) -> Tuple[int]:
            tokens = tokenizer.tokenize(sequence, add_special_tokens=False)
            res = [i for i, token in enumerate(tokens)
                   if token.endswith(configs.Data.mask_symbol)]
            return tuple(res)

        data = list(zip(sequences, [_get_mask_pattern_from_probing_sequence(s) for s in sequences]))
        return cls(sequences, tokenizer, ProbingParams(), data)

    def __init__(self,
                 sequences: List[str],
                 tokenizer: RobertaTokenizerFast,
                 params: Union[Params, ProbingParams],
                 data: Optional[List[Tuple[str, Tuple[int]]]] = None,
                 ):
        self._sequences = sequences
        self.tokenizer = tokenizer
        self.params = params

        if not self._sequences:  # empty devel or test set, for example
            print(f'WARNING: No sequences passed to {self}.')
            self.data = None
            return

        assert 0.0 <= self.params.leave_unmasked_prob <= 1.0
        assert 0.0 <= self.params.random_token_prob <= 1.0

        # weights for random token replacement
        weights = np.ones(self.tokenizer.vocab_size)
        weights[: len(self.tokenizer.all_special_tokens)] = 0
        self.weights = weights / weights.sum()

        print('Computing tokenized sequence lengths...', flush=True)
        self.tokenized_sequence_lengths, self.sequences = self._get_tokenized_sequence_lengths()
        print('Done')

        if not data:
            # create mask patterns + select which sequences are put in same batch (based on patterns)
            print('Creating new mask patterns...', flush=True)
            self.data = list(self._gen_sequences_and_mask_patterns())  # list of sequences, each with a mask pattern
            print('Done')

            # create batches of raw (non-vectorized data) in one of two ways:
            # 1) consecutive=true: sequences differing only in mask pattern are put in same batch.
            # 2) consecutive=false: sequences differing only in mask pattern are not put in same batch.
            # this is critical if training on data in order (e.g. age-order
            if not self.params.consecutive_masking:  # do not remove - use consecutive masking with "age-ordered"
                print('WARNING: Not using consecutive masking. Training data order is ignored.')
                random.shuffle(self.data)
        else:
            self.data = data

    def _gen_make_mask_patterns(self,
                                num_tokens_after_truncation: int,
                                ) -> Generator[Tuple[int], None, None]:
        """
        make a number of mask patterns that is as large as possible to the requested number.

        a mask_pattern is a tuple of 1 or more integers
        corresponding to the indices of a tokenized sequence that should be masked.

        notes:
        - pattern size is dynamically shortened if a tokenized sequence is smaller than mask_pattern_size.
        - num_mask_patterns is dynamically adjusted if number of possible patterns is smaller than num_mask_patterns.
        """
        random.seed(None)  # use different patterns across different runs

        pattern_size = min(self.params.mask_pattern_size, num_tokens_after_truncation)

        # sample patterns from population of all possible patterns
        all_mask_patterns = list(combinations(range(num_tokens_after_truncation), pattern_size))
        num_patterns = min(self.params.num_mask_patterns, len(all_mask_patterns))

        # generate mask patterns that are unique
        predetermined_patterns = iter(random.sample(all_mask_patterns, k=num_patterns))

        num_yielded = 0
        while num_yielded < num_patterns:

            if self.params.probabilistic_masking:
                if self.params.mask_probability == 'auto':
                    prob = self.params.mask_pattern_size / num_tokens_after_truncation
                elif isinstance(self.params.mask_probability, float) and 0 < self.params.mask_probability < 1:
                    prob = self.params.mask_probability
                else:
                    raise AttributeError('invalid arg to mask_probability')
                mask_pattern = tuple([i for i in range(num_tokens_after_truncation) if random.random() < prob])
            else:
                mask_pattern = next(predetermined_patterns)

            if mask_pattern:
                num_yielded += 1
            else:
                continue  # pattern can be empty when sampling probabilistically

            yield mask_pattern

    def _get_tokenized_sequence_lengths(self):
        """
        exclude sequences with too many tokens, if requested
        """

        tokenized_sequence_lengths = []
        sequences = []

        num_too_large = 0
        num_tokens_total = 0
        for s in self._sequences:
            tokens = self.tokenizer.tokenize(s, add_special_tokens=False)
            num_tokens = len(tokens)

            # exclude sequence if too many tokens
            num_tokens_and_special_symbols = num_tokens + 2
            if not self.params.allow_truncated_sentences and \
                    num_tokens_and_special_symbols > self.params.max_num_tokens_in_sequence:
                num_too_large += 1
                continue

            num_tokens_total += num_tokens
            num_tokens_after_truncation = min(self.params.max_num_tokens_in_sequence - 2,
                                              # -2 because we need to fit eos and bos symbols
                                              num_tokens)  # prevent masking of token in overflow region
            tokenized_sequence_lengths.append(num_tokens_after_truncation)
            sequences.append(s)

        if self.params.allow_truncated_sentences:
            print(f'Did not exclude sentences because truncated sentences are allowed.')
        else:
            print(f'Excluded {num_too_large} sequences with more than {self.params.max_num_tokens_in_sequence} tokens.')
        print(f'Mean number of tokens in sequence={num_tokens_total / len(sequences):.2f}',
              flush=True)

        return tokenized_sequence_lengths, sequences

    def _gen_sequences_and_mask_patterns(self) -> Generator[Tuple[str, Tuple[int]], None, None]:
        pbar = pyprind.ProgBar(len(self.sequences))
        for s, num_tokens_after_truncation in zip(self.sequences, self.tokenized_sequence_lengths):
            for mp in self._gen_make_mask_patterns(num_tokens_after_truncation):
                yield s, mp
            pbar.update()

    def _gen_data_chunks(self) -> Generator[Tuple[List[str], List[Tuple[int]]], None, None]:
        num_data = len(self.data)

        # sample data with or without replacement
        if self.params.sample_with_replacement:
            start_ids = np.random.randint(0, num_data - self.params.batch_size, size=num_data // self.params.batch_size)
        else:
            start_ids = range(0, num_data, self.params.batch_size)

        for start in start_ids:
            end = min(num_data, start + self.params.batch_size)
            sequences_in_batch, mask_patterns = zip(*self.data[start:end])
            yield list(sequences_in_batch), list(mask_patterns)

    @staticmethod
    def _make_mask_matrix(batch_shape: Tuple[int, int],
                          mask_patterns: List[Tuple[int]],
                          ) -> np.array:
        """
        return matrix specifying which tokens in a batch should be masked (but not necessarily replaced by mask symbol).

        notes:
        - mask_patterns is based on tokens without special symbols (eos, bos), so conversion must be done
        """
        res = np.zeros(batch_shape, dtype=np.bool)
        assert batch_shape[0] == len(mask_patterns)
        for row_id, mask_pattern in enumerate(mask_patterns):
            # a mask pattern may consist of zero, one, or more than one index (of a token to be masked)
            for mi in mask_pattern:
                col_id = mi + 1  # handle BOS symbol
                res[row_id, col_id] = True
        return res

    def mask_input_ids(self,
                       batch_encoding,
                       mask_patterns: List[Tuple[int]],
                       ) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None], torch.tensor], None, None]:

        batch_shape = batch_encoding.data['input_ids'].shape
        mask = self._make_mask_matrix(batch_shape, mask_patterns)

        # decide unmasking and random replacement
        rand_or_unmask_prob = self.params.random_token_prob + self.params.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(*batch_shape) < rand_or_unmask_prob)
            if self.params.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.params.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.params.leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(*batch_shape) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        # mask_insertion_matrix
        if unmask is not None:
            mask_insertion_matrix = mask ^ unmask  # XOR: True in mask will make True in mask_insertion False
        else:
            mask_insertion_matrix = mask

        # insert mask symbols - this has no effect during probing
        if np.any(mask_insertion_matrix):
            input_ids = np.where(mask_insertion_matrix,
                                 self.tokenizer.mask_token_id,
                                 batch_encoding.data['input_ids'])
        else:
            input_ids = np.copy(batch_encoding.data['input_ids'])

        # insert random tokens
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                input_ids[rand_mask] = np.random.choice(self.tokenizer.vocab_size, num_rand, p=self.weights)

        # x
        x = RobertaInput(input_ids=torch.tensor(input_ids),
                         attention_mask=torch.tensor(batch_encoding.data['attention_mask']),
                         position_ids=create_position_ids_from_input_ids(torch.tensor(batch_encoding.data['input_ids']),
                                                                         self.tokenizer.pad_token_id),
                         )

        # y
        if not mask_patterns:  # forced-choice probing
            y = None
        else:
            y = torch.tensor(batch_encoding.data['input_ids'][mask]).requires_grad_(False)

        # if self.params.leave_unmasked_prob > 0:
        #     print('mask')
        #     print(mask)
        #     print('unmask')
        #     print(unmask)
        #     print('mask_insertion_matrix')
        #     print(mask_insertion_matrix)
        #     print('rand_mask')
        #     print(rand_mask)
        #     print('input_ids - original')
        #     print(batch_encoding.data['input_ids'])
        #     print('input_ids - modified')
        #     print(input_ids)
        #     print()

        yield x, y, torch.tensor(mask)

    def __iter__(self) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None], torch.tensor], None, None]:

        """
        generate batches of vectorized data ready for training or probing.

        performs:
        - exclusion of long sequences
        - mask pattern creation
        - ordering of data
        - chunking
        - tokenization + vectorization
        - masking
        """

        for sequences_in_batch, mask_patterns in self._gen_data_chunks():
            
            batch_encoding = self.tokenizer.batch_encode_plus(sequences_in_batch,
                                                              is_pretokenized=False,
                                                              max_length=self.params.max_num_tokens_in_sequence,
                                                              padding=True,
                                                              truncation=True,
                                                              return_tensors='np')
            
            yield from self.mask_input_ids(batch_encoding, mask_patterns)
