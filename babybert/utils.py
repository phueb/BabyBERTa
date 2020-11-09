import random
import torch
from typing import Tuple, List, Optional, Generator, Union
import attr
from itertools import islice

from transformers import RobertaTokenizerFast, BatchEncoding
from transformers.modeling_roberta import create_position_ids_from_input_ids

from babybert import configs


@attr.s(slots=True, frozen=True)
class RobertaInput:
    input_ids: torch.tensor = attr.ib()
    attention_mask: torch.tensor = attr.ib()
    position_ids: torch.tensor = attr.ib()


def make_sequences(sentences: List[str],
                   num_sentences_per_input: int,
                   ) -> List[str]:

    gen = (bs for bs in sentences)

    # combine multiple sentences into 1 sequence
    res = []
    while True:
        sentences_in_sequence: List[str] = list(islice(gen, 0, num_sentences_per_input))
        if not sentences_in_sequence:
            break
        sequence = ' '.join(sentences_in_sequence)
        res.append(sequence)

        if len(res) % 100_000 == 0:
            print(f'Prepared {len(res):>12,} sequences', flush=True)

    print(f'Num total sequences={len(res):,}', flush=True)
    return res


def split(data: List[str],
          seed: int = 2) -> Tuple[List[str],
                                  List[str],
                                  List[str]]:

    print(f'Splitting data into train/devel/test sets...')

    random.seed(seed)

    train = []
    devel = []
    test = []

    for i in data:
        if random.choices([True, False],
                          weights=[configs.Data.train_prob, 1 - configs.Data.train_prob])[0]:
            train.append(i)
        else:
            if random.choices([True, False], weights=[0.5, 0.5])[0]:
                devel.append(i)
            else:
                test.append(i)

    print(f'num train sequences={len(train):,}', flush=True)
    print(f'num devel sequences={len(devel):,}', flush=True)
    print(f'num test  sequences={len(test):,}' , flush=True)

    return train, devel, test


def get_masked_indices(batch_encoding: BatchEncoding,
                       duplications: List[int]
                       ) -> Tuple[List[int], List[int]]:
    """
    guarantee that each duplicated sequence has mask in different location.
    inserts only 1 mask per sequence.
    does not mask padding, bos, or eos symbols.
    """
    row_indices = []
    col_indices = []
    encoding_id = 0
    row_id = 0
    for d in duplications:
        max_mask_location = sum(batch_encoding.encodings[encoding_id].attention_mask)  # does not consider eos, bos
        for col_id in random.sample(range(1, max_mask_location - 1), k=min(d, max_mask_location)):
            assert col_id <= configs.Data.max_sequence_length
            row_indices.append(row_id)
            col_indices.append(col_id)
            row_id += 1

        encoding_id += d  # d is used to get index for retrieving unique sequences in batch

    return row_indices, col_indices


def gen_batches(sequences: List[str],
                tokenizer: RobertaTokenizerFast,
                batch_size: int,
                num_masked: int,
                ) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None]], None, None]:

    if num_masked:
        assert batch_size % num_masked == 0
        num_unique_sequences_in_batch = batch_size // num_masked
    else:
        num_unique_sequences_in_batch = batch_size

    for start in range(0, len(sequences), num_unique_sequences_in_batch):

        # get unique sequences
        end = min(len(sequences), start + num_unique_sequences_in_batch)
        unique_sequences = sequences[start:end]

        # duplicate unique sequences in batch - each will get different mask
        sequences_in_batch = []
        duplications = []
        for s in unique_sequences:
            num_whole_words = len(s.split())
            num_duplicated = min(max(num_masked, 1), num_whole_words)
            duplications.append(num_duplicated)
            for _ in range(num_duplicated):
                sequences_in_batch.append(s)

        batch_encoding = tokenizer.batch_encode_plus(sequences_in_batch,
                                                     is_pretokenized=False,
                                                     max_length=configs.Data.max_sequence_length,
                                                     padding=True,
                                                     truncation=True,
                                                     return_tensors='pt')

        # mask - only once per sequence
        mask_pattern = torch.zeros_like(batch_encoding.data['input_ids'], dtype=torch.bool)
        if num_masked:
            row_indices, col_indices = get_masked_indices(batch_encoding, duplications)
            mask_pattern[row_indices, col_indices] = 1
            assert torch.sum(mask_pattern) == len(mask_pattern)
        input_ids_with_mask = torch.where(mask_pattern,
                                          torch.tensor(tokenizer.mask_token_id),
                                          batch_encoding.data['input_ids'])

        # encode sequences -> x
        x = RobertaInput(input_ids=input_ids_with_mask,
                         attention_mask=batch_encoding.data['attention_mask'],
                         position_ids=create_position_ids_from_input_ids(batch_encoding.data['input_ids'],
                                                                         tokenizer.pad_token_id),
                         )

        # encode labels -> y
        if not num_masked:  # when probing
            y = None
        else:
            y = batch_encoding.data['input_ids'].clone().detach().requires_grad_(False)[mask_pattern]

        yield x, y


def forward_mlm(model, mask_token_id, loss_fct, x, y):
    output = model(**{k: v.to('cuda') for k, v in attr.asdict(x).items()})
    logits_3d = output[0]
    logits_2d = logits_3d.view(-1, model.config.vocab_size)
    bool_1d = x.input_ids.view(-1) == mask_token_id
    logits_for_masked_words = logits_2d[bool_1d]
    labels = y.view(-1).cuda()
    loss = loss_fct(logits_for_masked_words,  # [num masks in batch, vocab size]
                    labels)  # [num masks in batch]

    return loss
