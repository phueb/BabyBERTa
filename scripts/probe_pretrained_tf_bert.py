from transformers import BertTokenizerFast, BertConfig, BertForPreTraining
import torch

from babybert import configs
from babybert.probing import do_probing


INCLUDE_PUNCTUATION = True

if __name__ == '__main__':

    # for each model

    for architecture_path in (configs.Dirs.root / 'pretrained_models').glob('*'):

        print(f'Entering {architecture_path}')

        # paths
        bert_config_path = architecture_path / 'bert_config.json'
        vocab_path = architecture_path / 'vocab.txt'

        # make tokenizer for tokenizing test sentences
        tokenizer = BertTokenizerFast(vocab_path, do_lower_case=configs.Data.lowercase_input, do_basic_tokenize=False)

        for path_to_bin in architecture_path.glob('*.bin'):

            # make bert model
            print()
            print('============================================')
            config = BertConfig.from_json_file(bert_config_path)
            print(f'Building PyTorch model from configuration in {bert_config_path}')
            model = BertForPreTraining(config)
            bin_file = configs.Dirs.root / 'pretrained_models' / path_to_bin
            state_dict = torch.load(bin_file)
            model.load_state_dict(state_dict)
            model.cuda(0)
            print(f'Num parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

            step = path_to_bin.name.split('_')[-2]
            rep_name = path_to_bin.name.split('_')[-3]
            save_path = configs.Dirs.probing_results / architecture_path.name / rep_name / 'saves'

            assert configs.Dirs.probing_sentences.exists()

            # for each probing task
            for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):

                do_probing(save_path, sentences_path, model, tokenizer, step, INCLUDE_PUNCTUATION)




