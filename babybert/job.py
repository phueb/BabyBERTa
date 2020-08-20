import time
import numpy as np
import pandas as pd
import attr
from pathlib import Path
import torch
import random
from itertools import chain

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.util import move_optimizer_to_cuda

from transformers import WordpieceTokenizer
from transformers import BertModel, BertConfig
from transformers import AdamW

from babybert import configs
from babybert.io import load_utterances_from_file
from babybert.io import load_propositions_from_file
from babybert.io import load_vocab
from babybert.utils import evaluate_pp, split
from babybert.probing import predict_open_ended


@attr.s
class Params(object):
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    embedding_dropout = attr.ib(validator=attr.validators.instance_of(float))
    num_layers = attr.ib(validator=attr.validators.instance_of(int))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    num_attention_heads = attr.ib(validator=attr.validators.instance_of(int))
    intermediate_size = attr.ib(validator=attr.validators.instance_of(int))
    num_mlm_epochs = attr.ib(validator=attr.validators.instance_of(int))
    srl_probability = attr.ib(validator=attr.validators.instance_of(float))
    srl_interleaved = attr.ib(validator=attr.validators.instance_of(bool))
    num_masked = attr.ib(validator=attr.validators.instance_of(int))
    vocab_size = attr.ib(validator=attr.validators.instance_of(int))
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

    #  paths
    project_path = Path(param2val['project_path'])
    save_path = Path(param2val['save_path'])
    srl_eval_path = project_path / 'perl' / 'srl-eval.pl'
    data_path_mlm = project_path / 'data' / 'training' / f'{params.corpus_name}_mlm.txt'
    data_path_train_srl = project_path / 'data' / 'training' / f'{params.corpus_name}_no-dev_srl.txt'
    data_path_devel_srl = project_path / 'data' / 'training' / f'human-based-2018_srl.txt'
    data_path_test_srl = project_path / 'data' / 'training' / f'human-based-2008_srl.txt'
    childes_vocab_path = project_path / 'data' / f'{params.corpus_name}_vocab.txt'
    google_vocab_path = project_path / 'data' / 'bert-base-uncased-vocab.txt'  # to get word pieces
    probing_path = project_path / 'data' / 'probing'

    if not probing_path.is_dir():  # when not using Ludwig
        probing_path = configs.Dirs.local_probing_path

    # prepare save_path - this must be done when job is executed locally (not on Ludwig worker)
    save_path = Path(param2val['save_path'])
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # word-piece tokenizer - defines input vocabulary
    print(f'Loading vocab with google_vocab_rule={params.google_vocab_rule}...')
    vocab = load_vocab(childes_vocab_path, google_vocab_path, params.vocab_size, params.google_vocab_rule)
    assert vocab['[PAD]'] == 0  # AllenNLP expects this
    assert vocab['[UNK]'] == 1  # AllenNLP expects this
    assert vocab['[CLS]'] == 2
    assert vocab['[SEP]'] == 3
    assert vocab['[MASK]'] == 4
    wordpiece_tokenizer = WordpieceTokenizer(vocab, unk_token='[UNK]')  # does not perform lower-casing
    print(f'Number of types in word-piece tokenizer={len(vocab):,}\n', flush=True)
    # note: but not all Google wordpieces are necessarily used (and are therefore not in Allen NLP vocab)

    # load utterances for MLM task
    utterances = load_utterances_from_file(data_path_mlm, allow_discard=True)
    train_utterances, devel_utterances, test_utterances = split(utterances)

    # load propositions for SLR task
    propositions = load_propositions_from_file(data_path_train_srl)
    train_propositions, devel_propositions, test_propositions = split(propositions)
    if data_path_devel_srl.is_file():  # use human-annotated data as devel split
        print(f'Using {data_path_devel_srl.name} as SRL devel split')
        devel_propositions = load_propositions_from_file(data_path_devel_srl)
    if data_path_test_srl.is_file():  # use human-annotated data as test split
        print(f'Using {data_path_test_srl.name} as SRL test split')
        test_propositions = load_propositions_from_file(data_path_test_srl)

    # converters handle conversion from text to instances
    converter_mlm = ConverterMLM(params.num_masked, wordpiece_tokenizer)
    converter_srl = ConverterSRL(params.num_masked, wordpiece_tokenizer)

    # get vocab for BERT heads
    # note: Allen NLP vocab holds labels, wordpiece_tokenizer.vocab holds input tokens
    # what from_instances() does:
    # 1. it iterates over all instances, and all fields, and all token indexers
    # 2. the token indexer is used to update vocabulary count, skipping words whose text_id is already set
    # 4. a PADDING and MASK symbol are added to 'tokens' namespace resulting in vocab size of 2
    # input tokens are not indexed, as they are already indexed by bert tokenizer vocab.
    # this ensures that the model is built with inputs for all vocab words,
    # such that words that occur only in LM or SRL task can still be input

    # make instances once - this allows iterating multiple times (required when num_epochs > 1)
    train_instances_mlm = converter_mlm.make_instances(train_utterances)
    devel_instances_mlm = converter_mlm.make_instances(devel_utterances)
    test_instances_mlm = converter_mlm.make_instances(test_utterances)
    train_instances_srl = converter_srl.make_instances(train_propositions)
    devel_instances_srl = converter_srl.make_instances(devel_propositions)
    test_instances_srl = converter_srl.make_instances(test_propositions)
    all_instances_mlm = chain(train_instances_mlm, devel_instances_mlm, test_instances_mlm)
    all_instances_srl = chain(train_instances_srl, devel_instances_srl, test_instances_srl)

    # make SRL vocab (for SRL labels) from all instances
    effective_vocab_srl = Vocabulary.from_instances(all_instances_srl)

    # BERT
    print('Preparing Multi-task BERT...')
    bert_config = BertConfig(vocab_size=len(wordpiece_tokenizer.vocab),  # was 32K
                             hidden_size=params.hidden_size,  # was 768
                             num_hidden_layers=params.num_layers,  # was 12
                             num_attention_heads=params.num_attention_heads,  # was 12
                             intermediate_size=params.intermediate_size,    # was 3072
                             )
    bert_model = BertModel(config=bert_config)
    # Multi-tasking BERT

    mt_bert = MTBert(id2tag_wp_mlm={i: t for t, i in wordpiece_tokenizer.vocab.items()},
                     id2tag_wp_srl=effective_vocab_srl.get_index_to_token_vocabulary('labels'),
                     bert_model=bert_model: BertForPreTraining,
                     embedding_dropout=params.embedding_dropout)
    mt_bert.cuda()
    num_params = sum(p.numel() for p in mt_bert.parameters() if p.requires_grad)
    print('Number of parameters: {:,}\n'.format(num_params), flush=True)

    # optimizers
    optimizer_mlm = AdamW(mt_bert.parameters(), lr=params.lr, correct_bias=False)
    optimizer_srl = AdamW(mt_bert.parameters(), lr=params.lr, correct_bias=False)
    move_optimizer_to_cuda(optimizer_mlm)
    move_optimizer_to_cuda(optimizer_srl)

    # batching
    bucket_batcher_mlm = BucketIterator(batch_size=params.batch_size, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher_srl = BucketIterator(batch_size=params.batch_size, sorting_keys=[('tokens', "num_tokens")])
    # bucket_batcher_mlm.index_with(effective_vocab_mlm)
    bucket_batcher_srl.index_with(effective_vocab_srl)

    # big batcher to speed evaluation - 1024 is too big
    large_batch_size = 256
    bucket_batcher_mlm_large = BucketIterator(batch_size=large_batch_size, sorting_keys=[('tokens', "num_tokens")])
    bucket_batcher_srl_large = BucketIterator(batch_size=large_batch_size, sorting_keys=[('tokens', "num_tokens")])
    # bucket_batcher_mlm_large.index_with(effective_vocab_mlm)
    bucket_batcher_srl_large.index_with(effective_vocab_srl)

    # init performance collection
    name2xy = {
        'devel_pps': [],
        'devel_f1s': [],
    }

    # init
    evaluated_steps_srl = []
    evaluated_steps_mlm = []
    train_start = time.time()
    loss_mlm = None
    no_mlm_batches = False
    step_mlm = 0
    step_srl = 0
    step_global = 0
    is_evaluated_at_current_step = False
    is_first_time_in_loop = True

    # generators
    train_generator_mlm = bucket_batcher_mlm(train_instances_mlm, num_epochs=params.num_mlm_epochs)
    train_generator_srl = bucket_batcher_srl(train_instances_srl, num_epochs=None)  # infinite generator

    # max step
    print('Counting number of training batches...')
    num_train_mlm_batches = bucket_batcher_mlm.get_num_batches(train_instances_mlm)
    num_train_srl_batches = bucket_batcher_srl.get_num_batches(train_instances_srl)
    print(f'num_train_mlm_batches={num_train_mlm_batches:>9,}')
    print(f'num_train_srl_batches={num_train_srl_batches:>9,}')  # but is an infinite generator
    # max step does not take into consideration number of unique SRL batches because it does not vary with num_masked.
    # the SRL batcher is infinite, and yields a batch with probability = srl_probability when interleaved = True,
    # or stop when max_step is reached when interleaved = False
    max_step = num_train_mlm_batches + (params.srl_probability * num_train_mlm_batches)
    print(f'Will stop training at global step={max_step:,}')
    print(flush=True)

    while step_global < max_step:

        # ####################################################################### TRAINING

        if not is_first_time_in_loop:  # do not influence first evaluation by training on first batch
            mt_bert.train()

            # masked language modeling task
            try:
                batch_mlm = next(train_generator_mlm)
            except StopIteration:
                if params.srl_interleaved:
                    break
                else:
                    no_mlm_batches = True
            else:
                loss_mlm = mt_bert.train_on_batch('mlm', batch_mlm, optimizer_mlm)
                step_mlm += 1

            # semantic role labeling task
            if params.srl_interleaved:
                if random.random() < params.srl_probability:
                    batch_srl = next(train_generator_srl)
                    mt_bert.train_on_batch('srl', batch_srl, optimizer_srl)
                    step_srl += 1
            elif no_mlm_batches:
                batch_srl = next(train_generator_srl)
                mt_bert.train_on_batch('srl', batch_srl, optimizer_srl)
                step_srl += 1

        is_first_time_in_loop = False
        step_global = step_mlm + step_srl

        # ####################################################################### EVALUATION

        # eval MLM
        if step_mlm % configs.Eval.interval == 0 and step_mlm not in evaluated_steps_mlm:
            evaluated_steps_mlm.append(step_mlm)
            is_evaluated_at_current_step = True
            mt_bert.eval()

            # evaluate devel perplexity
            devel_generator_mlm = bucket_batcher_mlm_large(devel_instances_mlm, num_epochs=1)
            devel_pp = evaluate_pp(mt_bert, devel_generator_mlm)

            name2xy['devel_pps'].append((step_mlm, devel_pp))
            print(f'devel-pp={devel_pp}', flush=True)

            # MLM test sentences
            if configs.Eval.test_sentences:
                test_generator_mlm = bucket_batcher_mlm_large(test_instances_mlm, num_epochs=1)
                out_path = save_path / f'test_split_mlm_results_{step_mlm}.txt'
                predict_open_ended(mt_bert, test_generator_mlm, out_path)

            # probing - test sentences for specific syntactic tasks
            skip_probing = step_global == 0 and not configs.Eval.probe_at_step_zero
            if not skip_probing:
                evaluate_on_tasks(probing_path, converter_mlm, bucket_batcher_mlm_large, save_path, mt_bert, step_mlm)

        # eval SRL
        if step_srl % configs.Eval.interval == 0 and step_srl not in evaluated_steps_srl:
            evaluated_steps_srl.append(step_srl)
            is_evaluated_at_current_step = True
            mt_bert.eval()

            # evaluate devel f1
            devel_generator_srl = bucket_batcher_srl_large(devel_instances_srl, num_epochs=1)
            devel_f1 = evaluate_model_on_f1(mt_bert, srl_eval_path, devel_generator_srl)

            name2xy['devel_f1s'].append((step_srl, devel_f1))
            print(f'devel-f1={devel_f1}', flush=True)

        # console
        if is_evaluated_at_current_step or step_global % configs.Training.feedback_interval == 0:
            min_elapsed = (time.time() - train_start) // 60
            pp = torch.exp(loss_mlm) if loss_mlm is not None else np.nan
            print(f'step MLM={step_mlm:>9,} | step SRL={step_srl:>9,} | step global={step_global:>9,}\n'
                  f'pp={pp :2.4f} \n'
                  f'total minutes elapsed={min_elapsed:<3}\n', flush=True)
            is_evaluated_at_current_step = False

    # ####################################################################### CLEANUP

    # evaluate train perplexity
    if configs.Eval.train_split:
        generator_mlm = bucket_batcher_mlm_large(train_instances_mlm, num_epochs=1)
        train_pp = evaluate_pp(mt_bert, generator_mlm)
    else:
        train_pp = np.nan
    print(f'train-pp={train_pp}', flush=True)

    # evaluate train f1
    if configs.Eval.train_split:
        generator_srl = bucket_batcher_srl_large(train_instances_srl, num_epochs=1)
        train_f1 = evaluate_model_on_f1(mt_bert, srl_eval_path, generator_srl, print_tag_metrics=True)
    else:
        train_f1 = np.nan
    print(f'train-f1={train_f1}', flush=True)

    # put train-pp and train-f1 evaluated at last step into pandas Series
    s1 = pd.Series([train_pp], index=[step_global])
    s1.name = 'train_pp'
    s2 = pd.Series([train_f1], index=[step_global])
    s2.name = 'train_f1'

    # MLM test sentences
    if configs.Eval.test_sentences:
        test_generator_mlm = bucket_batcher_mlm(test_instances_mlm, num_epochs=1)
        out_path = save_path / f'test_split_mlm_results_{step_mlm}.txt'
        predict_open_ended(mt_bert, test_generator_mlm, out_path)

    if configs.Eval.probe_at_end:
        evaluate_on_tasks(probing_path, converter_mlm, bucket_batcher_mlm_large, save_path, mt_bert, step_mlm)

    # return performance as pandas Series
    series_list = [s1, s2]
    for name, xy in name2xy.items():
        print(f'Making pandas series with name={name} and length={len(xy)}', flush=True)
        x, y = zip(*xy)
        s = pd.Series(y, index=x)
        s.name = name
        series_list.append(s)

    print('Reached end of babybert.job.main', flush=True)

    return series_list
