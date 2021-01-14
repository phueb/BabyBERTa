from typing import List, Tuple, Generator, Union

import torch
from transformers import BatchEncoding, RobertaTokenizerFast
from transformers.modeling_roberta import create_position_ids_from_input_ids

from babybert import configs
from babybert.utils import RobertaInput


def get_masked_indices(batch_encoding: BatchEncoding,
                       mask_patterns: List[Tuple[int]],
                       ) -> Tuple[List[int], List[int]]:
    """
    get row and column indices for making the matrix used to add mask symbols to input matrix.

    notes:
    - mask_pattern is based on tokens without special symbols (eos, bos), so conversion must be done
    """
    row_indices = []
    col_indices = []
    assert len(batch_encoding.encodings) == len(mask_patterns)
    for row_id, (encoding, mask_pattern) in enumerate(zip(batch_encoding.encodings, mask_patterns)):
        # a mask pattern may consist of more than one index (of a token to be masked)
        for mi in mask_pattern:
            col_id = mi + 1  # handle BOS symbol
            row_indices.append(row_id)
            col_indices.append(col_id)

    return row_indices, col_indices


def tokenize_and_mask(sequences_in_batch: List[str],
                      mask_patterns: List[Tuple[int]],
                      tokenizer: RobertaTokenizerFast,
                      ) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None]], None, None]:

    batch_encoding = tokenizer.batch_encode_plus(sequences_in_batch,
                                                 is_pretokenized=False,
                                                 max_length=configs.Data.max_num_tokens_in_sequence,
                                                 padding=True,
                                                 truncation=True,
                                                 return_tensors='pt')

    mask_pattern_size = len(mask_patterns[0])

    # add mask symbols to input ids, if not probing
    mask_matrix = torch.zeros_like(batch_encoding.data['input_ids'], dtype=torch.bool)
    if mask_pattern_size > 0:  # otherwise, probing
        row_indices, col_indices = get_masked_indices(batch_encoding, mask_patterns)
        mask_matrix[row_indices, col_indices] = 1
        # check masking
        num_masks = torch.sum(mask_matrix)
        num_expected_masks = len(mask_patterns) * mask_pattern_size
        assert num_masks == num_expected_masks
    input_ids_with_mask = torch.where(mask_matrix,
                                      torch.tensor(tokenizer.mask_token_id),
                                      batch_encoding.data['input_ids'])

    # x
    x = RobertaInput(input_ids=input_ids_with_mask,
                     attention_mask=batch_encoding.data['attention_mask'],
                     position_ids=create_position_ids_from_input_ids(batch_encoding.data['input_ids'],
                                                                     tokenizer.pad_token_id),
                     )

    # y
    if mask_pattern_size == 0:  # there are no mask patterns when probing
        y = None
    else:
        y = batch_encoding.data['input_ids'].clone().detach().requires_grad_(False)[mask_matrix]

    yield x, y