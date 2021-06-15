from transformers.models.roberta import RobertaTokenizer
from transformers import AutoTokenizer

from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

from babyberta import configs
from babyberta.utils import load_tokenizer

# TODO do not load the tokenizer like this:
# 1) it does not add prefix_space
# 2) it does not separate punctuation
# tokenizer = RobertaTokenizerFast.from_pretrained(configs.Dirs.saved_models / 'BabyBERTa_AO-CHILDES', from_slow=False)


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
print(tokenizer.decode(tokenizer.encode('Philip wanted to show you an example of tokenization.')).split())

tokenizer = AutoTokenizer.from_pretrained(configs.Dirs.saved_models / 'BabyBERTa_AO-CHILDES',
                                          # trim_offsets=True,  # does not help
                                          add_prefix_space=True
                                          )
# tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)  # does not help
# tokenizer.post_processor = TemplateProcessing(
#         single="<s> $A </s>",
#         pair=None,
#         special_tokens=[("<s>", tokenizer.bos_token_id), ("</s>", tokenizer.eos_token_id)],
#     )  # does not help
print(tokenizer.decode(tokenizer.encode('Philip wanted to show you an example of tokenization.')).split())

path_tokenizer_config = configs.Dirs.tokenizers / 'a-a-w-w-w-8192.json'
tokenizer = load_tokenizer(path_tokenizer_config, max_num_tokens_in_sequence=128)
print(tokenizer.encode('Philip wanted to show you an example of tokenization.').tokens)
