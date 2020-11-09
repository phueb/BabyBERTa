import random
import torch
from typing import Tuple, List, Optional, Generator, Union
import attr
from itertools import islice

from transformers import RobertaTokenizerFast
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


def gen_batches(sequences: List[str],
                tokenizer: RobertaTokenizerFast,
                batch_size: int,
                insert_masks: bool = True,
                ) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None]], None, None]:

    for start in range(0, len(sequences), batch_size):
        end = min(len(sequences), start + batch_size)

        encoding = tokenizer.batch_encode_plus(sequences[start:end],
                                               is_pretokenized=False,
                                               max_length=configs.Data.max_sequence_length,
                                               padding=True,
                                               truncation=True,
                                               return_tensors='pt')

        # mask - only once per sequence
        mask_pattern = torch.zeros_like(encoding.data['input_ids'], dtype=torch.bool)
        if insert_masks:
            mask_pattern[:, 2] = 1  # todo test deterministic masking
            assert torch.sum(mask_pattern) == len(mask_pattern)
        input_ids_with_mask = torch.where(mask_pattern,
                                          torch.tensor(tokenizer.mask_token_id),
                                          encoding.data['input_ids'])

        # encode sequences -> x
        x = RobertaInput(input_ids=input_ids_with_mask,
                         attention_mask=encoding.data['attention_mask'],
                         position_ids=create_position_ids_from_input_ids(encoding.data['input_ids'],
                                                                         tokenizer.pad_token_id),
                         )

        # encode labels -> y
        if not insert_masks:  # when probing
            y = None
        else:
            y = torch.tensor(encoding.data['input_ids'][mask_pattern],
                             device='cuda',
                             dtype=torch.long,
                             requires_grad=False)

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
