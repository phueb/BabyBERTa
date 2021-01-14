import time
import numpy as np
import pandas as pd
import attr
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss

from transformers import RobertaTokenizerFast
from transformers import BertForPreTraining, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from babybert import configs
from babybert.io import load_sentences_from_file
from babybert.utils import split, make_sequences, forward_mlm
from babybert.probing import do_probing
from babybert.batcher import Batcher, gen_batches


@attr.s
class Params(object):
    # data
    consecutive_masking = attr.ib(validator=attr.validators.instance_of(bool))
    num_sentences_per_input = attr.ib(validator=attr.validators.instance_of(int))
    training_order = attr.ib(validator=attr.validators.instance_of(str))
    include_punctuation = attr.ib(validator=attr.validators.instance_of(bool))
    allow_truncated_sentences = attr.ib(validator=attr.validators.instance_of(bool))
    num_mask_patterns = attr.ib(validator=attr.validators.instance_of(int))
    mask_pattern_size = attr.ib(validator=attr.validators.instance_of(int))
    corpus_name = attr.ib(validator=attr.validators.instance_of(str))
    bbpe = attr.ib(validator=attr.validators.instance_of(str))

    # training
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    num_epochs = attr.ib(validator=attr.validators.instance_of(int))
    num_warmup_steps = attr.ib(validator=attr.validators.instance_of(int))

    # model
    num_layers = attr.ib(validator=attr.validators.instance_of(int))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    num_attention_heads = attr.ib(validator=attr.validators.instance_of(int))
    intermediate_size = attr.ib(validator=attr.validators.instance_of(int))

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'project_path', 'save_path']}
        return cls(**kwargs)


def main(param2val):

    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    #  paths to data
    project_path = Path(param2val['project_path'])
    data_path_mlm = project_path / 'data' / 'corpora' / f'{params.corpus_name}.txt'

    # probing path - contains probing sentences
    probing_path = configs.Dirs.probing_sentences
    if not probing_path.exists():
        raise FileNotFoundError(f'Path to probing sentences does not exist: {probing_path}.'
                                'Probing sentences can be downloaded from github.com/phueb/Zorro/sentences')

    # save_path - locations where probing results are saved
    save_path = Path(param2val['save_path'])
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # B-BPE tokenizer - defines input vocabulary
    tokenizer = RobertaTokenizerFast(vocab_file=str(project_path / 'data' / 'tokenizers' / params.bbpe / 'vocab.json'),
                                     merges_file=str(project_path / 'data' / 'tokenizers' / params.bbpe / 'merges.txt'),
                                     add_prefix_space=configs.Data.add_prefix_space)

    # load text data
    sentences = load_sentences_from_file(data_path_mlm,
                                         training_order=params.training_order,
                                         include_punctuation=params.include_punctuation,
                                         allow_discard=True)
    all_sequences = make_sequences(sentences, params.num_sentences_per_input)
    train_sequences, devel_sequences, test_sequences = split(all_sequences)

    # BabyBERT
    print('Preparing BabyBERT...')
    bert_config = BertConfig(vocab_size=tokenizer.vocab_size,
                             hidden_size=params.hidden_size,
                             num_hidden_layers=params.num_layers,
                             num_attention_heads=params.num_attention_heads,
                             intermediate_size=params.intermediate_size,
                             )
    model = BertForPreTraining(config=bert_config)  # same as Roberta
    print('Number of parameters: {:,}'.format(model.num_parameters()), flush=True)
    model.cuda(0)

    # count number of steps in train ing data
    print('Counting training batches...', flush=True)
    max_step = len(list(Batcher(train_sequences,
                                tokenizer,
                                params.batch_size,
                                params.num_mask_patterns,
                                params.mask_pattern_size,
                                params.allow_truncated_sentences).gen_batch_sized_chunks(
        params.consecutive_masking))) * params.num_epochs
    print(f'max step={max_step:,}', flush=True)

    # optimizer + lr schedule
    optimizer = AdamW(model.parameters(), lr=params.lr, correct_bias=False)  # does not implement lr scheduling
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=params.num_warmup_steps,
                                                num_training_steps=max_step)

    # init performance collection
    name2xy = {}

    # init
    loss_fct = CrossEntropyLoss()
    train_start = time.time()
    loss = None
    step = 0
    is_evaluated_at_current_step = False
    is_first_time_in_loop = True
    eval_batch_size = configs.Eval.batch_size // params.num_sentences_per_input  # reduce chance of CUDA memory error

    # train + eval loop
    for epoch_id in range(params.num_epochs):
        for x, y in gen_batches(train_sequences, tokenizer, params):

            if not is_first_time_in_loop:  # do not influence first evaluation by training on first batch
                # forward
                model.train()
                loss = forward_mlm(model, tokenizer.mask_token_id, loss_fct, x, y)
                # backward + update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # otherwise only punctuation is predicted
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  # needed ?
                model.zero_grad()
                step += 1

            is_first_time_in_loop = False

            # eval
            if step % configs.Eval.interval == 0:
                is_evaluated_at_current_step = True

                # pp
                if configs.Data.train_prob < 1.0:  # if there are eval and test data
                    model.eval()
                    for sequences, name in zip([devel_sequences], ['devel']):
                        print(f'Computing {name} pp...', flush=True)
                        pp_sum = 0
                        num_steps = 0
                        for x, y in gen_batches(sequences, tokenizer, params, batch_size=eval_batch_size):
                            loss = forward_mlm(model, tokenizer.mask_token_id, loss_fct, x, y)
                            pp = torch.exp(loss).detach().cpu().numpy().item()
                            pp_sum += pp
                            num_steps += 1
                            model.zero_grad()
                        pp = pp_sum / num_steps
                        name2xy.setdefault(f'{name}_pps', []).append((step, pp))
                        print(f'{name} pp={pp}', flush=True)

                # probing - test sentences for specific syntactic tasks
                for sentences_path in probing_path.rglob('*.txt'):
                    do_probing(save_path, sentences_path, tokenizer, model, step, params.include_punctuation)

                if max_step - step < configs.Eval.interval:  # no point in continuing training
                    print('Detected last eval step. Exiting training loop', flush=True)
                    break

            # console
            if is_evaluated_at_current_step or step % configs.Training.feedback_interval == 0:
                min_elapsed = (time.time() - train_start) // 60
                pp = torch.exp(loss) if loss is not None else np.nan
                print(f'epoch={epoch_id + 1:>3,}/{params.num_epochs} step={step:>9,}/{max_step:>9,}\n'
                      f'pp={pp :2.4f} \n'
                      f'lr={scheduler.get_lr()[0]} \n'
                      f'total minutes elapsed={min_elapsed:<3}\n', flush=True)
                is_evaluated_at_current_step = False

    # prepare collected data for returning to Ludwig
    performance_curves = []
    for name, xy in name2xy.items():
        print(f'Making pandas series with name={name} and length={len(xy)}', flush=True)
        x, y = zip(*xy)
        s = pd.Series(y, index=x)
        s.name = name
        performance_curves.append(s)

    print('Reached end of babybert.job.main', flush=True)

    return performance_curves
