import random
import torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from torch.nn import CrossEntropyLoss
from typing import Tuple, List
import attr
from itertools import islice
from pathlib import Path

from babybert import configs


loss_fct = CrossEntropyLoss()


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


def forward_mlm(model,
                mask_matrix: torch.bool,  # mask_matrix is 2D bool array specifying which tokens to predict
                x: torch.tensor,
                y: torch.tensor,
                ) -> torch.tensor:
    output = model(**{k: v.to('cuda') for k, v in attr.asdict(x).items()})
    logits_3d = output['logits']
    logits_2d = logits_3d.view(-1, model.config.vocab_size)
    bool_1d = mask_matrix.view(-1)
    logits_for_masked_words = logits_2d[bool_1d]
    labels = y.view(-1).cuda()
    loss = loss_fct(logits_for_masked_words,  # [num masks in batch, vocab size]
                    labels)  # [num masks in batch]

    return loss


def load_tokenizer(params,
                   project_path: Path,
                   ) -> Tokenizer:
    if params.bbpe == 'gpt2_bpe':
        raise NotImplementedError
    json_fn = f'{params.bbpe}.json'

    tokenizer = Tokenizer.from_file(str(project_path / 'data' / 'tokenizers' / json_fn))
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair=None,
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('<pad>'), pad_token='<pad>')
    tokenizer.enable_truncation(max_length=params.max_num_tokens_in_sequence)
    return tokenizer
