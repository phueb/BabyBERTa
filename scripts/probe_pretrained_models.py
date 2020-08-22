from transformers import BertTokenizer, BertConfig, BertForPreTraining
import torch

from babybert import configs
from babybert.probing import do_probing


INCLUDE_PUNCTUATION = True

if __name__ == '__main__':

    # make tokenizer for tokenizing test sentences
    vocab_path = configs.Dirs.root / 'pretrained_models' / 'vocab_new.txt'
    tokenizer = BertTokenizer(vocab_path, do_lower_case=False, do_basic_tokenize=False)

    # for each model
    for path_to_bin in (configs.Dirs.root / 'pretrained_models').glob('*/*.bin'):

        # load weights
        architecture_name = path_to_bin.parent.name
        bert_config_path = configs.Dirs.root / 'pretrained_models' / architecture_name / 'bert_config.json'
        bin_file = configs.Dirs.root / 'pretrained_models' / path_to_bin

        # make bert model
        print()
        print('============================================')
        config = BertConfig.from_json_file(bert_config_path)
        print(f'Building PyTorch model from configuration in {bert_config_path}')
        model = BertForPreTraining(config)
        state_dict = torch.load(bin_file)
        model.load_state_dict(state_dict)
        model.cuda(0)
        print(f'Num parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

        step = path_to_bin.name.split('_')[-2]
        rep_name = path_to_bin.name.split('_')[-3]
        save_path = configs.Dirs.probing_results / architecture_name / rep_name / 'saves'

        # for each probing task
        for task_name in configs.Eval.probing_names:
            do_probing(task_name, save_path, configs.Dirs.probing_sentences, tokenizer, model, step,
                       INCLUDE_PUNCTUATION)




