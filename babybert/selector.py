from typing import List, Tuple, Generator
import random
from transformers import RobertaTokenizerFast


class Selector:

    def __init__(self,
                 sequences: List[str],
                 tokenizer: RobertaTokenizerFast,
                 batch_size: int,
                 num_masked: int,
                 ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_masked = num_masked

    def _make_masked_locations(self, sequence: str):
        tokenized = self.tokenizer.tokenize(sequence)
        res = random.sample(range(len(tokenized)), k=min(self.num_masked, len(tokenized)))
        return res

    def _data(self) -> Generator[Tuple[str, int], None, None]:
        for s in self.sequences:
            for ml in self._make_masked_locations(s):
                yield s, ml

    def gen_batch_sized_chunks(self,
                               consecutive_masking: bool,
                               ) -> Generator[Tuple[List[str], List[int]], None, None]:
        """
        used to select sequences in one of two ways:
        1) consecutive=true: sequences differing only in mask location are put in same batch.
        2) consecutive=false: sequences differing only in mask location are not put in same batch.
        """
        print('Computing masked locations...', flush=True)
        data = list(self._data())
        if not consecutive_masking:
            random.shuffle(data)
        print('Done')

        # split data into batch-sized chunks
        for start in range(0, len(data), self.batch_size):
            end = min(len(data), start + self.batch_size)
            sequences_in_batch, masked_locations = zip(*data[start:end])
            yield list(sequences_in_batch), list(masked_locations)
