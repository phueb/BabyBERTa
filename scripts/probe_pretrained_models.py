from pathlib import Path
from typing import Dict
from collections import OrderedDict
from transformers import WordpieceTokenizer, BertConfig, BertForPreTraining
import torch

from babybert import configs
from babybert.io import load_utterances_from_file
from babybert.io import save_open_ended_predictions, save_forced_choice_predictions
from babybert.probing import predict_open_ended, predict_forced_choice


def make_w2id_from_vocab_file() -> Dict[str, int]:
    res = OrderedDict()
    index = 0
    vocab_path = configs.Dirs.root / 'pretrained_models' / 'vocab_new.txt'
    for token in vocab_path.open().read().split('\n'):
        if token in res:
            # happens for symbols
            continue
        res[token] = index
        index += 1

    print(f'Loaded vocab with {len(res):,} words')

    return res


if __name__ == '__main__':

    # path to probing data - probing data can be found at https://github.com/phueb/Babeval/tree/master/sentences
    probing_path = Path().home() / 'Babeval_phueb' / 'sentences'

    # TODO [PAD] is not necessarily at index 0, and [MASK] is not necessarily at index 4
    # make wordpiece tokenizer for tokenizing test sentences
    w2id = make_w2id_from_vocab_file()
    tokenizer = WordpieceTokenizer(w2id, unk_token='[UNK]')

    # for each model
    for path_to_bin in (configs.Dirs.root / 'pretrained_models').glob('*/*.bin'):
        architecture_name = path_to_bin.parent
        bert_config_path = configs.Dirs.root / 'pretrained_models' / architecture_name / 'bert_config.json'
        bin_file = configs.Dirs.root / 'pretrained_models' / path_to_bin

        # load bert model
        config = BertConfig.from_json_file(bert_config_path)
        print(f'Building PyTorch model from configuration in {bert_config_path}')
        model = BertForPreTraining(config)
        state_dict = torch.load(bin_file)
        model.load_state_dict(state_dict)
        model.cuda(0)
        print(f'Num parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

        # for each probing task
        for probing_task_name in configs.Eval.probing_names:
            for task_type in ['forced_choice', 'open_ended']:

                # prepare data - data is expected to be located on shared drive
                probing_data_path_mlm = probing_path / task_type / f'{probing_task_name}.txt'
                if not probing_data_path_mlm.exists():
                    print(f'WARNING: {probing_data_path_mlm} does not exist', flush=True)
                    continue
                print(f'Starting probing with task={probing_task_name}', flush=True)
                probing_utterances_mlm = load_utterances_from_file(probing_data_path_mlm)

                # batch data
                batches = None  # TODO

                # prepare out path
                save_path = configs.Dirs.root / 'pretrained_models' / architecture_name / 'saves'
                step = path_to_bin.name.split('_')[-2]  # TODO test
                probing_results_path = save_path / task_type / f'probing_{probing_task_name}_results_{step}.txt'
                if not probing_results_path.parent.exists():
                    probing_results_path.parent.mkdir(exist_ok=True)

                # do inference on forced-choice task
                if task_type == 'forced_choice':
                    sentences_in, cross_entropies = predict_forced_choice(model, batches)
                    save_forced_choice_predictions(sentences_in, cross_entropies, probing_results_path)

                # do inference on open_ended task
                elif task_type == 'open_ended':
                    sentences_in, sentences_out = predict_open_ended(model, batches)
                    save_open_ended_predictions(sentences_in, sentences_out, probing_results_path)

                else:
                    raise AttributeError('Invalid arg to "task_type".')




