import time
import numpy as np
import pandas as pd
import attr
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss

from transformers.modeling_roberta import create_position_ids_from_input_ids
from transformers.tokenization_roberta import AddedToken
from transformers import RobertaTokenizerFast
from transformers import BertForPreTraining, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from babybert import configs
from babybert.io import load_utterances_from_file
from babybert.utils import evaluate_pp, split, gen_batches_with_labels, do_masking, combine, concatenate_utterances
from babybert.probing import do_probing


@attr.s
class Params(object):
    # data
    num_utterances_per_input = attr.ib(validator=attr.validators.instance_of(int))
    training_order = attr.ib(validator=attr.validators.instance_of(str))
    include_punctuation = attr.ib(validator=attr.validators.instance_of(bool))
    num_masked = attr.ib(validator=attr.validators.instance_of(int))
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
                                'Probing sentences can be downloaded from github.com/phueb/Babeval/sentences')

    # save_path - locations where probing results are saved
    save_path = Path(param2val['save_path'])
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # B-BPE tokenizer - defines input vocabulary
    tokenizer = RobertaTokenizerFast(vocab_file=str(project_path / 'data' / 'tokenizers' / params.bbpe / 'vocab.json'),
                                     merges_file=str(project_path / 'data' / 'tokenizers' / params.bbpe / 'merges.txt'))
    tokenizer.add_special_tokens({'mask_token': AddedToken('[MASK]', lstrip=True)})

    # load utterances for MLM + do masking
    utterances = load_utterances_from_file(data_path_mlm,
                                           training_order=params.training_order,
                                           include_punctuation=params.include_punctuation,
                                           allow_discard=True)
    # each is a tuple with elements: (masked_utterances, masked_word)
    train_data, devel_data, test_data = split(combine(do_masking(utterances,
                                                                 params.num_masked,
                                                                 ),
                                                      params.num_utterances_per_input))

    # BabyBERT
    print('Preparing BabyBERT...')
    bert_config = BertConfig(vocab_size=tokenizer.vocab_size,
                             hidden_size=params.hidden_size,
                             num_hidden_layers=params.num_layers,
                             num_attention_heads=params.num_attention_heads,
                             intermediate_size=params.intermediate_size,
                             )
    model = BertForPreTraining(config=bert_config)  # same as Roberta
    print('Number of parameters: {:,}\n'.format(model.num_parameters()), flush=True)
    model.cuda(0)

    # optimizer + lr schedule
    optimizer = AdamW(model.parameters(), lr=params.lr, correct_bias=False)  # does not implement lr scheduling
    max_step = len(train_data) // params.batch_size * params.num_epochs
    print(f'max step={max_step:,}')
    exit()
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=params.num_warmup_steps,
                                                num_training_steps=max_step)

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
    eval_batch_size = configs.Eval.batch_size // params.num_utterances_per_input  # reduce chance of CUDA memory error

    # train + eval loop
    for epoch_id in range(params.num_epochs):  # TODO test epochs
        for train_batch in gen_batches_with_labels(train_data, params.batch_size):


            if not is_first_time_in_loop:  # do not influence first evaluation by training on first batch
                model.train()

                # possibly, concatenate multiple utterances into 1 input sequence
                masked_sequences, masked_words = concatenate_utterances(train_batch)

                # forward MLM
                batch = tokenizer.batch_encode_plus([' '.join(s) for s in masked_sequences],
                                                    padding=True,
                                                    return_tensors='pt')
                position_ids = create_position_ids_from_input_ids(batch.data['input_ids'], tokenizer.pad_token_id)

                output = model(**batch.to('cuda'), position_ids=position_ids.to('cuda'))
                logits_3d = output[0]
                logits_2d = logits_3d.view(-1, model.config.vocab_size)

                # loss
                masked_word_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(masked_words),
                                                     device='cuda',
                                                     dtype=torch.long,
                                                     requires_grad=False)
                logits_for_masked_words = logits_2d[batch.data['input_ids'].view(-1) == tokenizer.mask_token_id]
                loss = loss_fct(logits_for_masked_words,  # [num masks in batch, vocab size]
                                masked_word_token_ids.view(-1))  # [num masks in batch]

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
            if step % configs.Eval.interval == 0 and step not in evaluated_steps:
                evaluated_steps.append(step)
                is_evaluated_at_current_step = True
                model.eval()

                # pp
                skip_pp = step == 0 and not configs.Eval.eval_pp_at_step_zero
                if not skip_pp:
                    print('Computing train pp...', flush=True)
                    train_pp = evaluate_pp(model, tokenizer, train_data, eval_batch_size)
                    print('Computing devel pp...', flush=True)
                    devel_pp = evaluate_pp(model, tokenizer, devel_data, eval_batch_size)
                    name2xy['train_pps'].append((step, train_pp))
                    name2xy['devel_pps'].append((step, devel_pp))
                    print(f'train-pp={train_pp}', flush=True)
                    print(f'devel-pp={devel_pp}', flush=True)

                # probing - test sentences for specific syntactic tasks
                for sentences_path in probing_path.rglob('*.txt'):
                    do_probing(save_path, sentences_path, tokenizer, model, step, params.include_punctuation)

                if max_step - step < configs.Eval.interval: # no point in continuing training
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
