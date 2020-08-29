import random
import torch
from torch.nn import CrossEntropyLoss
from typing import Tuple, List, Iterator, Generator

from transformers import BertForPreTraining, BertTokenizer

from babybert import configs


def do_masking(utterances: List[List[str]],
               num_masked: int,
               ) -> Generator[Tuple[List[str], str], None, None]:
    """
    Prepare input for masked language modeling:
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


def combine(data: Generator[Tuple[List[str], str], None, None],
            num_utterances_per_input: int,
            ) -> Generator[List[Tuple[List[str], str]], None, None]:
    """
    combines multiple (utterance, label) pairs into a list of multiple such pairs.
    this controls how many utterances make up a single input sequence.
    in the original BERT implementation, each input sequence contains multiple sentences.
    """

    print(f'Number of utterances per input sequence={num_utterances_per_input}')

    safe_guard = 0
    while True:

        safe_guard += 1
        if safe_guard > 10_000_000:
            raise SystemExit('SAFE GUARD')

        # get a slice of the data
        try:
            combined = [next(data) for _ in range(num_utterances_per_input)]
        except StopIteration:
            print(f'Num combination steps={safe_guard:,}')
            return

        yield combined


def split(data: Generator[List[Tuple[List[str], str]], None, None],
          seed: int = 2) -> Tuple[List[List[Tuple[List[str], str]]],
                                  List[List[Tuple[List[str], str]]],
                                  List[List[Tuple[List[str], str]]]]:

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

    print(f'num train={len(train):,}')
    print(f'num devel={len(devel):,}')
    print(f'num test ={len(test):,}')

    return train, devel, test


def evaluate_pp(model: BertForPreTraining,
                tokenizer: BertTokenizer,
                data: List[List[Tuple[List[str], str]]],
                ) -> float:
    model.eval()

    loss_fct = CrossEntropyLoss()

    pp_sum = torch.zeros(size=(1,)).cuda()
    num_steps = 0
    for batch in gen_batches_with_labels(data, 512):
        masked_utterances, masked_words = concatenate_utterances(batch)

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


def gen_batches_without_labels(utterances: List[List[str]],
                               batch_size: int) -> Generator[List[List[str]], None, None]:
    """used during probing, where labels are not needed"""
    for start in range(0, len(utterances), batch_size):
        end = min(len(utterances), start + batch_size)
        yield utterances[start:end]


def gen_batches_with_labels(data: List[List[Tuple[List[str], str]]],
                            batch_size: int) -> Generator[List[List[Tuple[List[str], str]]], None, None]:
    """used for training, where labels are required"""
    for start in range(0, len(data), batch_size):
        end = min(len(data), start + batch_size)
        yield data[start:end]


def concatenate_utterances(train_batch: List[List[Tuple[List[str], str]]],
                           ) -> Tuple[List[List[str], List[str]]]:
    """
    re-arrange data structure of batch for BERT tokenizer, which expects data of type List[List[str]]

    Each item in the incoming data structure (a list) is a list of (masked utterance, masked word) pairs.
    This data structure needs to be converted into 2 outgoing data structures:
    1. A batch (list) where each item is a sequence composed of 1, or multiple concatenated, utterances
    2. A batch (list) of masked words

    convert from
    [(masked utterance1, masked word1),(masked utterance2, masked word2), ..], ..]
    to
    [concatenated utterances1, concatenated utterances2, ..] and [masked word1, masked word2, ...]
    """
    masked_utterances, masked_words = [], []

    for combination in train_batch:
        us, ws = zip(*combination)
        masked_utterances.append([])
        for u, w in zip(us, ws):
            masked_utterances[-1].extend(u)
            masked_words.append(w)

    return masked_utterances, masked_words
