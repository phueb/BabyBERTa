"""
Example code for loading BabyBERTa pre-trained on 5M words of child-directed input.

Careful: BabyBERTa uses a custom tokenizer which was trained with add_prefix_space=True
"""

from transformers.models.roberta import RobertaTokenizer
from transformers import AutoTokenizer

from babyberta import configs
from babyberta.utils import load_tokenizer

##################################################
# roberta-base - RobertaTokenizer.from_pretrained
##################################################

tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                             add_prefix_space=False)
print(tokenizer.tokenize('Philip wanted to show you an example of tokenization.'))

##################################################
# babyberta - load_tokenizer
##################################################

path_tokenizer_config = configs.Dirs.tokenizers / 'babyberta.json'
tokenizer = load_tokenizer(path_tokenizer_config, max_input_length=128)
print(tokenizer.encode('Philip wanted to show you an example of tokenization.', add_special_tokens=False).tokens)

##################################################
# babyberta - AutoTokenizer.from_pretrained
##################################################

tokenizer = AutoTokenizer.from_pretrained('saved_models/BabyBERTa_AO-CHILDES',
                                          add_prefix_space=True,  # this must be added to produce intended behavior
                                          )
print(tokenizer.tokenize('Philip wanted to show you an example of tokenization.'))
