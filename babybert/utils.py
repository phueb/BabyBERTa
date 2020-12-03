import random
import torch
from typing import Tuple, List, Optional, Generator, Union
import attr
from itertools import islice

from transformers import RobertaTokenizerFast, BatchEncoding
from transformers.modeling_roberta import create_position_ids_from_input_ids

from babybert import configs
from babybert.selector import Selector


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
                       masked_locations: List[int]
                       ) -> Tuple[List[int], List[int]]:
    """
    inserts only 1 mask per sequence.
    does not mask padding, bos, or eos symbols.
    """
    row_indices = []
    col_indices = []
    assert len(batch_encoding.encodings) == len(masked_locations)
    for row_id, (encoding, ml) in enumerate(zip(batch_encoding.encodings,
                                                masked_locations)):
        col_id = ml + 1  # because of BOS symbol
        max_mask_location = sum(encoding.attention_mask) - 1  # because of EOS symbol
        assert col_id < max_mask_location
        row_indices.append(row_id)
        col_indices.append(col_id)

    return row_indices, col_indices


def tokenize_and_mask(sequences_in_batch: List[str],
                      masked_locations: List[int],
                      tokenizer: RobertaTokenizerFast,
                      probing: bool,
                      ) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None]], None, None]:

    batch_encoding = tokenizer.batch_encode_plus(sequences_in_batch,
                                                 is_pretokenized=False,
                                                 max_length=configs.Data.max_sequence_length,
                                                 padding=True,
                                                 truncation=True,
                                                 return_tensors='pt')

    # mask - only once per sequence
    mask_pattern = torch.zeros_like(batch_encoding.data['input_ids'], dtype=torch.bool)
    if not probing:
        row_indices, col_indices = get_masked_indices(batch_encoding, masked_locations)
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
    if probing:  # when probing
        y = None
    else:
        y = batch_encoding.data['input_ids'].clone().detach().requires_grad_(False)[mask_pattern]

    yield x, y


def gen_batches(sequences: List[str],
                tokenizer: RobertaTokenizerFast,
                batch_size: int,
                consecutive_masking: bool,  # if true, duplicated sequences are in same batch
                num_masked: Optional[int] = None,  # number of times to duplicate a sequence (each with different mask)
                probing: Optional[bool] = None,
                ) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None]], None, None]:

    # must specify one of the two
    if num_masked is None and probing is None:
        raise ValueError('Must specify  either num_masked or probing.')

    if probing:
        assert num_masked is None
        num_masked = 1  # probing sentences are not duplicated
    elif num_masked:
        assert probing is None
        probing = False

    # selector selects which sequences are put in same batch (based on masked locations)
    selector = Selector(sequences, tokenizer, batch_size, num_masked)

    for sequences_in_batch, masked_locations in selector.gen_batch_sized_chunks(consecutive_masking):
        yield from tokenize_and_mask(sequences_in_batch, masked_locations, tokenizer, probing)


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
