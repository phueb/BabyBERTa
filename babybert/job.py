import time
import numpy as np
import pandas as pd
import attr
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss

from transformers import BertTokenizer
from transformers import BertForPreTraining, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from babybert import configs
from babybert.io import load_utterances_from_file
from babybert.memory import set_memory_limit
from babybert.io import make_vocab
from babybert.utils import evaluate_pp, split, gen_batches_with_labels, do_masking
from babybert.probing import do_probing


@attr.s
class Params(object):
    include_punctuation = attr.ib(validator=attr.validators.instance_of(bool))
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    training_order = attr.ib(validator=attr.validators.instance_of(str))
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

    set_memory_limit(prop=0.9)

    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    #  paths to data
    project_path = Path(param2val['project_path'])
    data_path_mlm = project_path / 'data' / 'corpora' / f'{params.corpus_name}.txt'
    childes_vocab_path = project_path / 'data' / 'vocabulary' / f'{params.corpus_name}_vocab.txt'
    google_vocab_path = project_path / 'data' / 'vocabulary' / 'bert-base-uncased-vocab.txt'  # to get word pieces

    # probing path - contains probing sentences
    probing_path = configs.Dirs.probing_sentences
    if not probing_path.exists():
        raise FileNotFoundError(f'Path to probing sentences does not exist: {probing_path}.'
                                'Probing sentences can be downloaded from github.com/phueb/Babeval/sentences')

    # save_path - locations where probing results are saved
    save_path = Path(param2val['save_path'])
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # word-piece tokenizer - defines input vocabulary
    print(f'Loading vocab with google_vocab_rule={params.google_vocab_rule}...')
    vocab = make_vocab(childes_vocab_path, google_vocab_path, params.childes_vocab_size, params.google_vocab_rule)
    custom_vocab_path = project_path / 'data' / 'vocabulary' / 'effective_vocab.txt'
    custom_vocab_path.open('w').write('\n'.join(vocab))
    tokenizer = BertTokenizer(custom_vocab_path, do_lower_case=False, do_basic_tokenize=False)
    print(f'Number of types in word-piece tokenizer={len(vocab):,}\n', flush=True)

    # load utterances for MLM + do masking
    utterances = load_utterances_from_file(data_path_mlm,
                                           training_order=params.training_order,
                                           include_punctuation=params.include_punctuation,
                                           allow_discard=True)
    # each is a tuple with elements: (masked_utterances, masked_word)
    train_data, devel_data, test_data = split(do_masking(utterances, params.num_masked))

    # BERT
    print('Preparing BERT...')
    bert_config = BertConfig(vocab_size=len(tokenizer.vocab),  # was 32K
                             hidden_size=params.hidden_size,  # was 768
                             num_hidden_layers=params.num_layers,  # was 12
                             num_attention_heads=params.num_attention_heads,  # was 12
                             intermediate_size=params.intermediate_size,    # was 3072
                             )
    model = BertForPreTraining(config=bert_config)
    print('Number of parameters: {:,}\n'.format(model.num_parameters()), flush=True)
    model.cuda(0)

    # optimizer + lr schedule
    optimizer = AdamW(model.parameters(), lr=params.lr, correct_bias=False)
    max_step = len(train_data) // params.batch_size * params.num_epochs
    print(f'max step={max_step:,}')
    # get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10_000, num_training_steps=max_step)  # TODO

    # init performance collection
    name2xy = {
        'train_pps': [],
        'devel_pps': [],
    }

    # init
    loss_fct = CrossEntropyLoss()
    evaluated_steps = []
    train_start = time.time()
    loss = None
    step = 0
    is_evaluated_at_current_step = False
    is_first_time_in_loop = True

    # train + eval loop
    for epoch_id in range(params.num_epochs):  # TODO test epochs
        for train_batch in gen_batches_with_labels(train_data, params.batch_size):

            # print(optimizer.param_groups[0]["lr"])  # TODO test lr schedule

            if not is_first_time_in_loop:  # do not influence first evaluation by training on first batch
                model.train()
                masked_utterances, masked_words = zip(*train_batch)

                # forward MLM
                batch = tokenizer(masked_utterances,
                                  padding=True,
                                  return_tensors="pt",
                                  is_pretokenized=True)
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

                # backward + update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # otherwise only punctuation is predicted
                optimizer.step()
                optimizer.zero_grad()  # needed ?
                model.zero_grad()
                step += 1

            is_first_time_in_loop = False

            # eval
            if step % configs.Eval.interval == 0 and step not in evaluated_steps:
                evaluated_steps.append(step)
                is_evaluated_at_current_step = True
                model.eval()

                # pp
                skip_pp = step == 0 and not configs.Eval.eval_pp_at_step_zero
                if not skip_pp:
                    print('Computing train pp...', flush=True)
                    train_pp = evaluate_pp(model, tokenizer, train_data[:len(devel_data)])
                    print('Computing devel pp...', flush=True)
                    devel_pp = evaluate_pp(model, tokenizer, devel_data)
                    name2xy['train_pps'].append((step, train_pp))
                    name2xy['devel_pps'].append((step, devel_pp))
                    print(f'train-pp={train_pp}', flush=True)
                    print(f'devel-pp={devel_pp}', flush=True)

                # probing - test sentences for specific syntactic tasks
                for task_name in configs.Eval.probing_names:
                    do_probing(task_name, save_path, probing_path, tokenizer, model, step, params.include_punctuation)

                if max_step - step < configs.Eval.interval: # no point in continuing training
                    print('Detected last eval step. Exiting training loop', flush=True)
                    break

            # console
            if is_evaluated_at_current_step or step % configs.Training.feedback_interval == 0:
                min_elapsed = (time.time() - train_start) // 60
                pp = torch.exp(loss) if loss is not None else np.nan
                print(f'epoch={epoch_id + 1:>3,}/{params.num_epochs} step={step:>9,}/{max_step:>9,}\n'
                      f'pp={pp :2.4f} \n'
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
