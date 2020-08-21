import random
import torch
from typing import Iterator, Optional, List

from transformers import BertForPreTraining

from babybert import configs


def evaluate_pp(model: BertForPreTraining,
                batches: Iterator,
                ) -> float:
    model.eval()

    pp_sum = torch.zeros(size=(1,)).cuda()
    num_steps = 0
    for step, batch in enumerate(batches):

        # get predictions
        with torch.no_grad():
            output_dict = model(task='mlm', **batch)  # input is dict[str, tensor]

        pp = torch.exp(output_dict['loss'])
        pp_sum += pp
        num_steps += 1

    return pp_sum.cpu().numpy().item() / num_steps


def split(data: List, seed: int = 2):

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

    print(f'num train={len(train):,}')
    print(f'num devel={len(devel):,}')
    print(f'num test ={len(test):,}')

    return train, devel, test


def gen_batches(utterances: List[List[str]], bs: int):
    for start in range(0, len(utterances), bs):
        end = min(len(utterances), start + bs)
        yield utterances[start:end]
