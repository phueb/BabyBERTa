"""
This script checks that the two available methods for loading a BabyBERTa tokenizer produce identical results.
"""
from pathlib import Path
from transformers.models.roberta import RobertaTokenizerFast

from babyberta import configs
from babyberta.io import load_tokenizer

PATH_TO_SENTENCES = Path('/home/ph/Zorro/sentences/babyberta')

##################################################
# babyberta - load_tokenizer
##################################################

path_tokenizer_config = configs.Dirs.tokenizers / 'babyberta.json'
tokenizer1 = load_tokenizer(path_tokenizer_config, max_input_length=128)

##################################################
# babyberta - AutoTokenizer.from_pretrained
##################################################

tokenizer2 = RobertaTokenizerFast.from_pretrained('../saved_models/BabyBERTa_AO-CHILDES',
                                                  add_prefix_space=True,  # this must be added for intended behavior
                                                  )


for paradigm_path in PATH_TO_SENTENCES.glob('*.txt'):
    print(f'Checking tokenization with sentences in {paradigm_path}')
    for line in paradigm_path.read_text().split('\n'):

        masked_id = 3
        line = ' '.join([w if n != masked_id else '<mask>' for n, w in enumerate(line.split())])

        res1 = tokenizer1.encode(line, add_special_tokens=False).tokens
        res2 = tokenizer2.tokenize(line)

        print(res1)
        print(res2)

        assert res1 == res2

