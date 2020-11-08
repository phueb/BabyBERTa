import random
import torch
from torch.nn import CrossEntropyLoss
from typing import Tuple, List, Optional, Generator, Union
import attr
from itertools import islice

from transformers import RobertaTokenizerFast
from transformers.modeling_roberta import create_position_ids_from_input_ids
from transformers import BertForPreTraining

from babybert import configs


@attr.s(slots=True, frozen=True)
class BBPESequence:
    sequence: List[str] = attr.ib()  # possibly multiple sentences in 1 sequence
    labels: Optional[Tuple[str]] = attr.ib(default=None)  # masked words - always exactly one per sentence


@attr.s(slots=True, frozen=True)
class RobertaInput:
    input_ids = attr.ib()
    attention_mask = attr.ib()
    position_ids = attr.ib()


def make_bbpe_sequences(sentences: List[str],
                        tokenizer: RobertaTokenizerFast,
                        num_masked: int,
                        num_sentences_per_input: int,
                        ) -> List[BBPESequence]:

    print('Making BBPE sequences...')

    # tokenize with BBPE
    bbpe_sentences = [tokenizer.tokenize(s) for s in sentences]

    #  optional masking
    if num_masked == 0:  # probing
        print('Not adding masks')
        gen = ((bs, None) for bs in bbpe_sentences)
    else:
        gen = gen_masking(bbpe_sentences, num_masked)

    # combine multiple sentences into 1 sequence
    res = []
    while True:
        tmp = list(zip(*islice(gen, 0, num_sentences_per_input)))
        if not tmp:
            break
        if not len(tmp) == 2: print(tmp)
        masked_sentences, labels = tmp
        sequence = [w for s in masked_sentences for w in s]
        bbpe_sequence = BBPESequence(sequence=sequence, labels=labels)
        res.append(bbpe_sequence)

        if len(res) % 100_000 == 0:
            print(f'Prepared {len(res):>12,} sequences', flush=True)

    print(f'Num total BBPE sequences={len(res):,}', flush=True)
    return res


def gen_masking(bbpe_sentences: List[List[str]],
                num_masked: int,
                ) -> Generator[Tuple[List[str], str], None, None]:
    """
    Duplicate sentences, each time adding 1 mask in different position.
    contrary to original BERT,
    only 1 location is masked per sentence, and masked locations are replaced by:
    100% MASK, 0% random, 0% original.
    """

    for bs in bbpe_sentences:
        for loc in random.sample(range(len(bs)), k=min(num_masked, len(bs))):
            masked_sentence = [w if n != loc else configs.Data.mask_symbol for n, w in enumerate(bs)]
            masked_word = bs[loc]
            yield masked_sentence, masked_word


def split(data: List[BBPESequence],
          seed: int = 2) -> Tuple[List[BBPESequence],
                                  List[BBPESequence],
                                  List[BBPESequence]]:

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


def gen_batches(bbpe_sequences: List[BBPESequence],
                tokenizer: RobertaTokenizerFast,
                batch_size: int,
                no_labels: bool = False,
                ) -> Generator[Tuple[RobertaInput, Union[torch.LongTensor, None]], None, None]:

    for start in range(0, len(bbpe_sequences), batch_size):
        end = min(len(bbpe_sequences), start + batch_size)

        tokenized_sequences = []
        masked_words = []
        for bbpe_sequence in bbpe_sequences[start:end]:
            tokenized_sequences.append(bbpe_sequence.sequence)
            masked_words.extend(bbpe_sequence.labels)

        encoding = tokenizer.batch_encode_plus(tokenized_sequences,
                                               is_pretokenized=True,
                                               padding=True,
                                               return_tensors='pt').to('cuda')
        encoding['position_ids'] = create_position_ids_from_input_ids(encoding.data['input_ids'],
                                                                      tokenizer.pad_token_id)
        x = RobertaInput(**encoding)
        if no_labels:  # when probing
            y = None
        else:
            y = torch.tensor(tokenizer.convert_tokens_to_ids(masked_words),
                             device='cuda',
                             dtype=torch.long,
                             requires_grad=False)
        yield x, y


def evaluate_pp(model: BertForPreTraining,
                tokenizer: RobertaTokenizerFast,
                bbpe_sequences: List[BBPESequence],
                batch_size: int,
                ) -> float:
    model.eval()

    loss_fct = CrossEntropyLoss()

    pp_sum = torch.zeros(size=(1,)).cuda()
    num_steps = 0
    for x, y in gen_batches(bbpe_sequences, tokenizer, batch_size):

        with torch.no_grad():
            output = model(**attr.asdict(x))
            logits_3d = output[0]
            logits_2d = logits_3d.view(-1, model.config.vocab_size)
            logits_for_masked_words = logits_2d[x.input_ids.view(-1) == tokenizer.mask_token_id]
            loss = loss_fct(logits_for_masked_words,
                            y.view(-1))

        pp = torch.exp(loss)
        pp_sum += pp
        num_steps += 1

    return pp_sum.cpu().numpy().item() / num_steps
