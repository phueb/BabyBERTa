import random
import torch
from torch.nn import CrossEntropyLoss
from typing import Tuple, List, Iterator, Generator

from transformers import BertForPreTraining, BertTokenizer

from babybert import configs


def do_masking(utterances: List[List[str]],
               num_masked: int,
               ) -> Generator[Tuple[List[List[str]], List[str]], None, None]:
    """
    Prepare masked tokens inputs/labels for masked language modeling:
    contrary to original BERT,
    only 1 location is masked per utterance, and masked locations are replaced by:
    100% MASK, 0% random, 0% original.
    """

    print(f'Inserting masked symbols into utterances...')

    for u in utterances:
        for loc in random.sample(range(len(u)), k=min(num_masked, len(u))):
            masked_utterance = [w if n != loc else '[MASK]' for n, w in enumerate(u)]
            masked_word = u[loc]
            yield masked_utterance, masked_word


def evaluate_pp(model: BertForPreTraining,
                tokenizer: BertTokenizer,
                data: List[Tuple[List[List[str]], List[str]]],
                ) -> float:
    model.eval()

    loss_fct = CrossEntropyLoss()

    pp_sum = torch.zeros(size=(1,)).cuda()
    num_steps = 0
    for batch in gen_batches_with_labels(data, 512):
        masked_utterances, masked_words = zip(*batch)

        with torch.no_grad():
            batch = tokenizer(masked_utterances,
                              padding=True,
                              return_tensors="pt",
                              is_pretokenized=True,
                              return_attention_mask=True)

            # logits
            output = model(**batch.to('cuda'))
            logits_3d = output[0]
            logits_2d = logits_3d.view(-1, model.config.vocab_size)

            # loss
            masked_word_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(masked_words),
                                                 device='cuda',
                                                 dtype=torch.long,
                                                 requires_grad=False)
            logits_for_masked_words = logits_2d[batch.data['input_ids'].view(-1) == tokenizer.mask_token_id]
            loss = loss_fct(logits_for_masked_words,  # [batch size, vocab size]
                            masked_word_token_ids.view(-1))  # [batch size]

        pp = torch.exp(loss)
        pp_sum += pp
        num_steps += 1

    return pp_sum.cpu().numpy().item() / num_steps


def split(data: Iterator, seed: int = 2):

    print(f'Splitting utterances into train/devel/test sets...')

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


def gen_batches_without_labels(utterances: List[List[str]],
                               bs: int):
    """used during probing, where labels are not needed"""
    for start in range(0, len(utterances), bs):
        end = min(len(utterances), start + bs)
        yield utterances[start:end]


def gen_batches_with_labels(data: List[Tuple[List[List[str]], List[str]]],
                            bs: int):
    """used for training, where labels are required"""
    for start in range(0, len(data), bs):
        end = min(len(data), start + bs)
        yield data[start:end]
