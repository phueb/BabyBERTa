import random
import torch
from torch.nn import CrossEntropyLoss
from typing import Tuple, List, Dict
from itertools import islice

from babyberta import configs

loss_fct = CrossEntropyLoss()


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

    print(f'Num total sequences={len(res):,}', flush=True)
    return res


def split(data: List[str],
          seed: int = 2) -> Tuple[List[str],
                                  List[str]]:

    print(f'Splitting data into train/dev sets...')

    random.seed(seed)

    train = []
    devel = []

    for i in data:
        if random.choices([True, False],
                          weights=[configs.Data.train_prob, 1 - configs.Data.train_prob])[0]:
            train.append(i)
        else:
            devel.append(i)

    print(f'num train sequences={len(train):,}', flush=True)
    print(f'num devel sequences={len(devel):,}', flush=True)

    return train, devel


def forward_mlm(model,
                mask_matrix: torch.bool,  # mask_matrix is 2D bool array specifying which tokens to predict
                x: Dict[str, torch.tensor],
                y: torch.tensor,
                ) -> torch.tensor:
    output = model(**{k: v.to('cuda') for k, v in x.items()})
    logits_3d = output['logits']
    logits_2d = logits_3d.view(-1, model.config.vocab_size)
    bool_1d = mask_matrix.view(-1)
    logits_for_masked_words = logits_2d[bool_1d]
    labels = y.view(-1).cuda()
    loss = loss_fct(logits_for_masked_words,  # [num masks in batch, vocab size]
                    labels)  # [num masks in batch]

    return loss
