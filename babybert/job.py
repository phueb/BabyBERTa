import time
import numpy as np
import pandas as pd
import attr
from pathlib import Path
import torch

from transformers import WordpieceTokenizer
from transformers import BertForPreTraining, BertConfig
from transformers import AdamW

from babybert import configs
from babybert.io import load_utterances_from_file
from babybert.io import load_vocab
from babybert.utils import evaluate_pp, split
from babybert.probing import do_probing


@attr.s
class Params(object):
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    num_layers = attr.ib(validator=attr.validators.instance_of(int))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    num_attention_heads = attr.ib(validator=attr.validators.instance_of(int))
    intermediate_size = attr.ib(validator=attr.validators.instance_of(int))
    num_epochs = attr.ib(validator=attr.validators.instance_of(int))
    num_masked = attr.ib(validator=attr.validators.instance_of(int))
    childes_vocab_size = attr.ib(validator=attr.validators.instance_of(int))
    google_vocab_rule = attr.ib(validator=attr.validators.instance_of(str))
    corpus_name = attr.ib(validator=attr.validators.instance_of(str))

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
    childes_vocab_path = project_path / 'data' / 'vocabulary' / f'{params.corpus_name}_vocab.txt'
    google_vocab_path = project_path / 'data' / 'vocabulary' / 'bert-base-uncased-vocab.txt'  # to get word pieces
    probing_path = project_path / 'data' / 'probing'

    if not probing_path.is_dir():  # when not using Ludwig
        probing_path = configs.Dirs.local_probing_path

    # prepare save_path - this must be done when job is executed locally (not on Ludwig worker)
    save_path = Path(param2val['save_path'])
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # word-piece tokenizer - defines input vocabulary
    print(f'Loading vocab with google_vocab_rule={params.google_vocab_rule}...')
    vocab = load_vocab(childes_vocab_path, google_vocab_path, params.childes_vocab_size, params.google_vocab_rule)
    assert vocab['[PAD]'] == 0
    assert vocab['[UNK]'] == 1
    assert vocab['[CLS]'] == 2
    assert vocab['[SEP]'] == 3
    assert vocab['[MASK]'] == 4
    wordpiece_tokenizer = WordpieceTokenizer(vocab, unk_token='[UNK]')  # does not perform lower-casing
    print(f'Number of types in word-piece tokenizer={len(vocab):,}\n', flush=True)

    # load utterances for MLM task
    utterances = load_utterances_from_file(data_path_mlm, allow_discard=True)
    train_utterances, devel_utterances, test_utterances = split(utterances)

    # TODO batching

    # BERT
    print('Preparing BERT...')
    bert_config = BertConfig(vocab_size=len(wordpiece_tokenizer.vocab),  # was 32K
                             hidden_size=params.hidden_size,  # was 768
                             num_hidden_layers=params.num_layers,  # was 12
                             num_attention_heads=params.num_attention_heads,  # was 12
                             intermediate_size=params.intermediate_size,    # was 3072
                             )
    model = BertForPreTraining(config=bert_config)
    model.cuda(0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: {:,}\n'.format(num_params), flush=True)

    # optimizers
    optimizer = AdamW(model.parameters(), lr=params.lr, correct_bias=False)

    # init performance collection
    name2xy = {
        'train_pps': [],
        'devel_pps': [],
    }

    # init
    evaluated_steps = []
    train_start = time.time()
    loss_mlm = None
    step = 0
    is_evaluated_at_current_step = False
    is_first_time_in_loop = True

    # max step
    max_step = len(batches)

    for batch in batches:

        if not is_first_time_in_loop:  # do not influence first evaluation by training on first batch
            model.train()

            # masked language modeling task
            loss_mlm = model.train_on_batch('mlm', batch, optimizer)
            step += 1

        is_first_time_in_loop = False

        # eval MLM
        if step % configs.Eval.interval == 0 and step not in evaluated_steps:
            evaluated_steps.append(step)
            is_evaluated_at_current_step = True
            model.eval()

            # pp
            train_pp = evaluate_pp(model, generator_mlm)
            devel_pp = evaluate_pp(model, generator_mlm)
            name2xy['train_pp'].append((step, train_pp))
            name2xy['devel_pp'].append((step, devel_pp))

            print(f'train-pp={devel_pp}', flush=True)
            print(f'devel-pp={devel_pp}', flush=True)

            # probing - test sentences for specific syntactic tasks
            skip_probing = step == 0 and not configs.Eval.eval_at_step_zero
            if not skip_probing:
                for task_name in configs.Eval.probing_names:
                    do_probing(task_name, save_path, probing_path, model, step)

        # console
        if is_evaluated_at_current_step or step % configs.Training.feedback_interval == 0:
            min_elapsed = (time.time() - train_start) // 60
            pp = torch.exp(loss_mlm) if loss_mlm is not None else np.nan
            print(f'step global={step:>9,}/{max_step}\n'
                  f'pp={pp :2.4f} \n'
                  f'total minutes elapsed={min_elapsed:<3}\n', flush=True)
            is_evaluated_at_current_step = False

    if configs.Eval.eval_at_end:
        # pp
        train_pp = evaluate_pp(model, generator_mlm)
        devel_pp = evaluate_pp(model, generator_mlm)
        name2xy['train_pp'].append((step, train_pp))
        name2xy['devel_pp'].append((step, devel_pp))

        # probing tasks
        for task_name in configs.Eval.probing_names:
            do_probing(task_name, save_path, probing_path, model, step)

    performance_curves = []
    for name, xy in name2xy.items():
        print(f'Making pandas series with name={name} and length={len(xy)}', flush=True)
        x, y = zip(*xy)
        s = pd.Series(y, index=x)
        s.name = name
        performance_curves.append(s)

    print('Reached end of babybert.job.main', flush=True)

    return performance_curves
