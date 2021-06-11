from transformers.models.roberta import RobertaTokenizerFast

from babyberta import configs
from babyberta.utils import load_tokenizer

# TODO do not load the tokenizer like this
tokenizer = RobertaTokenizerFast.from_pretrained(configs.Dirs.saved_models / 'BabyBERTa_AO-CHILDES', from_slow=False)
print(tokenizer.decode(tokenizer.encode('Philip wanted to show you an example of tokenization.')))
# ['ph', 'il', 'ip', 'Ġwanted', 'Ġto', 'Ġshow', 'Ġyou', 'Ġan', 'Ġexample', 'Ġof', 'Ġto', 'ken', 'ization', '.']


path_tokenizer_config = configs.Dirs.tokenizers / 'a-a-w-w-w-8192.json'
tokenizer = load_tokenizer(path_tokenizer_config, max_num_tokens_in_sequence=128)
print(tokenizer.encode('Philip wanted to show you an example of tokenization.').tokens)
