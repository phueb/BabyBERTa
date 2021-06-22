"""
Example code for loading BabyBERTa pre-trained on 5M words of child-directed input.

Careful: BabyBERTa uses a custom tokenizer which was trained with add_prefix_space=True
"""

from transformers.models.roberta import RobertaTokenizerFast

##################################################
# roberta-base - RobertaTokenizer.from_pretrained
##################################################

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base',
                                                 add_prefix_space=False)
print(tokenizer.tokenize('Philip wanted to show you an example of tokenization.'))

##################################################
# babyberta - AutoTokenizer.from_pretrained
##################################################

tokenizer = RobertaTokenizerFast.from_pretrained('saved_models/BabyBERTa_AO-CHILDES',
                                                 add_prefix_space=True,  # must be added to produce intended behavior
                                                 )
print(tokenizer.tokenize('Philip wanted to show you an example of tokenization.'))
