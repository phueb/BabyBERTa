"""
Use MLM scoring method to evaluate models on BLiMP or Zorro.


"+5M" refers to AO-CHILDES
"+13M" refers to Wikipedia-1

WITHOUT capitalization of proper nouns, average accuracies on zorro:
RoBERTa-base+5M                  73.64
BabyBERTa+Wikipedia-1            73.82
BabyBERTa+AO-CHILDES+50Kvocab    77.11
BabyBERTa+AO-Newsela             77.31
RoBERTa-base+10M                 78.47
BabyBERTa+AO-CHILDES             78.93
RoBERTa-base+13M                 79.78
BabyBERTa+concatenated           86.5
RoBERTa-base+30B                 90.63

WITH capitalization of proper nouns, average accuracies on zorro:
RoBERTa-base+5M                  72.24 (down because proper nouns are lower-cased in training data  and tokenizer does not lower case during eval)
BabyBERTa+Wikipedia-1            73.82 (same)
BabyBERTa+AO-CHILDES+50Kvocab    75.82 (down)
BabyBERTa+AO-Newsela             77.31 (same)
RoBERTa-base+13M                 78.25 (down because proper nouns are lower-cased in training data and tokenizer does not lower case during eval)
BabyBERTa+AO-CHILDES             78.93 (same)
RoBERTa-base+10M                 79.55 (up)
BabyBERTa+concatenated           86.5  (same)
RoBERTa-base+30B                 91.13 (up)
"""
import shutil
from pathlib import Path

from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizerFast, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from fairseq.models.roberta import RobertaModel
from convert_fairseq_roberta import convert_fairseq_roberta_to_pytorch

from src.scoring import score_model_on_paradigm
from src.accuracy import calc_and_print_accuracy

from babyberta import configs
from babyberta.utils import load_tokenizer

DATA_NAME = 'zorro'  # zorro or blimp
LOWER_CASE = False  # this affects both zorro (proper nouns) and blimp

if DATA_NAME == 'blimp':
    num_paradigms = 67
elif DATA_NAME == 'zorro':
    num_paradigms = 23
else:
    raise AttributeError('Invalid "data_name".')

# BabyBERTa models trained by us
babyberta_models = [p.name for p in (configs.Dirs.root / 'saved_models').glob('*')]
# trained by others
huggingface_models = ['RoBERTa-base_10M', 'RoBERTa-base_30B']
# trained by us using fairseq - needs to be converted
fairseq_models = ['RoBERTa-base_5M', 'RoBERTa-base_13M']

model_names = babyberta_models + fairseq_models + huggingface_models

path_to_output_dir = Path(DATA_NAME) / 'output' / f'lower_case={LOWER_CASE}'

for model_name in model_names:

    path_model_output_dir = (path_to_output_dir / model_name)
    if path_model_output_dir.exists():
        if len(list(path_model_output_dir.glob('*.txt'))) == num_paradigms:
            print(f'Output already exists. Skipping evaluation of {model_name}.')
            continue
        else:
            shutil.rmtree(path_model_output_dir)

    if not path_model_output_dir.exists():
        path_model_output_dir.mkdir(parents=True)

    # BabyBERTa with 50K vocab and RobertaTokenizerFast
    if '50K' in model_name:
        model = RobertaForMaskedLM.from_pretrained(f'../saved_models/{model_name}')
        # this method of loading the tokenizer produces the same results as load_tokenizer()
        tokenizer = RobertaTokenizerFast.from_pretrained(f'../saved_models/{model_name}', from_slow=False)

    # BabyBERTa with tokenizers.Tokenizer
    elif model_name.startswith('BabyBERTa'):
        model = RobertaForMaskedLM.from_pretrained(f'../saved_models/{model_name}')

        # TODO this does not tokenize correctly and produces lower accuracy
        # tokenizer = RobertaTokenizerFast.from_pretrained(f'../saved_models/{model_name}', from_slow=False)

        path_tokenizer_config = configs.Dirs.tokenizers / 'babyberta.json'
        tokenizer = load_tokenizer(path_tokenizer_config, max_input_length=128)

    # pre-trained RoBERTa-base by Warstadt et al., 2020
    elif model_name == 'RoBERTa-base_10M':
        model = AutoModelForMaskedLM.from_pretrained("nyu-mll/roberta-base-10M-2")
        tokenizer = AutoTokenizer.from_pretrained("nyu-mll/roberta-base-10M-2")
        vocab = tokenizer.get_vocab()

    # pre-trained RoBERTa-base by Liu et al., 2019
    elif model_name == 'RoBERTa-base_30B':
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        vocab = tokenizer.get_vocab()

    # pre-trained by us using fairseq
    elif model_name in fairseq_models:
        if '5M' in model_name:
            data_size = '5M'  # AO-CHILDES
            bin_name = 'aochildes-data-bin'
        elif '13M' in model_name:
            data_size = '13M'  # Wikipedia-1
            bin_name = 'wikipedia1_new1_seg'
        else:
            raise AttributeError('Invalid data size for fairseq model.')
        path_model_data = configs.Dirs.root / 'fairseq_models' / f'fairseq_Roberta-base_{data_size}'
        model_ = RobertaModel.from_pretrained(model_name_or_path=str(path_model_data / '0'),
                                              checkpoint_file=str(path_model_data / '0' / 'checkpoint_last.pt'),
                                              data_name_or_path=str(path_model_data / bin_name),
                                              )
        model, tokenizer = convert_fairseq_roberta_to_pytorch(model_)

    else:
        raise AttributeError('Invalid arg to name')

    vocab = tokenizer.get_vocab()
    model.eval()
    model.cuda(0)

    assert path_model_output_dir.exists()

    # compute scores
    for path_paradigm in (configs.Dirs.mlm_scoring / DATA_NAME / 'data').glob('*.txt'):
        print(f"Scoring pairs in {path_paradigm} with {model_name}...")
        path_out_file = path_model_output_dir / path_paradigm.name
        score_model_on_paradigm(model, vocab, tokenizer, path_paradigm, path_out_file, lower_case=LOWER_CASE)

# compute accuracy
calc_and_print_accuracy(DATA_NAME, LOWER_CASE)

