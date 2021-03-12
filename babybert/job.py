import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from transformers.models.bert import BertForPreTraining, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from babybert import configs
from babybert.io import load_sentences_from_file
from babybert.params import Params
from babybert.utils import split, make_sequences, forward_mlm, load_tokenizer
from babybert.probing import do_probing
from babybert.dataset import DataSet


def main(param2val):

    import transformers

    assert transformers.__version__ == '4.3.3'
    assert torch.__version__ == '1.6.0+cu101'

    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    #  paths to data
    project_path = Path(param2val['project_path'])
    data_path = project_path / 'data' / 'corpora' / f'{params.corpus_name}.txt'

    # probing path - contains probing sentences
    probing_path = configs.Dirs.probing_sentences
    if not probing_path.exists():
        raise FileNotFoundError(f'Path to probing sentences does not exist: {probing_path}.'
                                'Probing sentences can be downloaded from github.com/phueb/Zorro/sentences')

    # save_path - locations where probing results are saved
    save_path = Path(param2val['save_path'])
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Byte-level BPE tokenizer
    tokenizer = load_tokenizer(params, project_path)
    vocab_size = len(tokenizer.get_vocab())
    print(f'Vocab size={vocab_size}')

    # load text data
    sentences = load_sentences_from_file(data_path,
                                         training_order=params.training_order,
                                         include_punctuation=params.include_punctuation,
                                         allow_discard=True)
    all_sequences = make_sequences(sentences, params.num_sentences_per_input)
    train_sequences, devel_sequences, test_sequences = split(all_sequences)

    # BabyBERT
    print('Preparing BabyBERT...')

    # roberta_config = RobertaConfig(vocab_size=vocab_size,
    #                                pad_token_id=tokenizer.token_to_id('<pad>'),
    #                                bos_token_id=tokenizer.token_to_id('<s>'),
    #                                eos_token_id=tokenizer.token_to_id('</s>'),
    #                                return_dict=True,
    #                                is_decoder=False,
    #                                is_encoder_decoder=False,
    #                                add_cross_attention=False,
    #                                max_position_embeddings=params.max_num_tokens_in_sequence,
    #                                hidden_size=params.hidden_size,
    #                                num_hidden_layers=params.num_layers,
    #                                num_attention_heads=params.num_attention_heads,
    #                                intermediate_size=params.intermediate_size,
    #                                initializer_range=params.initializer_range,
    #                                )
    # model = RobertaForMaskedLM(config=roberta_config)

    bert_config = BertConfig(vocab_size=vocab_size,
                             hidden_size=params.hidden_size,
                             num_hidden_layers=params.num_layers,
                             num_attention_heads=params.num_attention_heads,
                             intermediate_size=params.intermediate_size,
                             initializer_range=params.initializer_range,
                             )
    model = BertForPreTraining(config=bert_config)  # same as Roberta
    print('Number of parameters: {:,}'.format(model.num_parameters()), flush=True)
    model.cuda(0)

    train_dataset = DataSet(train_sequences, tokenizer, params)
    devel_dataset = DataSet(devel_sequences, tokenizer, params)
    test_dataset = DataSet(test_sequences, tokenizer, params)

    # count number of steps in training data
    max_step = len(train_dataset.data) // params.batch_size * params.num_epochs
    print(f'max step={max_step:,}', flush=True)

    # optimizer + lr schedule
    optimizer = AdamW(model.parameters(),
                      lr=params.lr,
                      weight_decay=params.weight_decay,
                      correct_bias=False)  # does not implement lr scheduling
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=params.num_warmup_steps,
                                                num_training_steps=max_step)

    # init
    name2xy = {}
    train_start = time.time()
    loss = None
    step = 0
    is_evaluated_at_current_step = False
    is_first_time_in_loop = True

    # train + eval loop
    for epoch_id in range(params.num_epochs):
        for x, y, mm in train_dataset:

            if not is_first_time_in_loop:  # do not influence first evaluation by training on first batch
                # forward
                model.train()
                loss = forward_mlm(model, mm, x, y)
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
                    for ds, name in zip([devel_dataset], ['devel']):
                        print(f'Computing {name} pp...', flush=True)
                        pp_sum = 0
                        num_steps = 0
                        for x_eval, y_eval, mm_eval in devel_dataset:
                            loss = forward_mlm(model, mm_eval, x_eval, y_eval)
                            pp = torch.exp(loss).detach().cpu().numpy().item()
                            pp_sum += pp
                            num_steps += 1
                            model.zero_grad()
                        pp = pp_sum / num_steps
                        name2xy.setdefault(f'{name}_pps', []).append((step, pp))
                        print(f'{name} pp={pp}', flush=True)

                # probing - test sentences for specific syntactic tasks
                for sentences_path in probing_path.rglob('*.txt'):
                    do_probing(save_path, sentences_path, model, tokenizer, step,
                               params.include_punctuation, params.score_with_mask)

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

