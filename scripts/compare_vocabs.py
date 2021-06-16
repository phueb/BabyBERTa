
"""
Count number of words in BabyBERTa vocab that are also in Roberta-base vocab
"""
import numpy as np
from pathlib import Path

from transformers.models.roberta import RobertaTokenizer

from babyberta import configs
from babyberta.utils import load_tokenizer

tokenizer_base = RobertaTokenizer.from_pretrained("roberta-base")


path_tokenizer_config = configs.Dirs.tokenizers / 'babyberta.json'
tokenizer_baby = load_tokenizer(path_tokenizer_config, max_input_length=128)


vocab_base = [v.strip('Ġ') for v in tokenizer_base.get_vocab()]
vocab_baby = [v.strip('Ġ') for v in tokenizer_baby.get_vocab()]

num_overlapping = 0
num_total = 0
for vw in vocab_baby:
    if vw in vocab_base:
        num_overlapping += 1
    num_total += 1

ratio = num_overlapping / num_total
print(f'{num_overlapping}/{num_total}={ratio:.2f} of BabyBERTa vocab items in Roberta-base vocab')


###########################


PATH_TO_SENTENCES = Path('/home/ph/Zorro/sentences/babyberta')
PATH_TO_STOPWORDS = Path('/home/ph/Zorro/data/external_words/')

stop_words = set((PATH_TO_STOPWORDS / "stopwords.txt").open().read().split())

# collect words in test suite
words_in_test_suite = set()
for paradigm_path in PATH_TO_SENTENCES.glob('*.txt'):
    print(paradigm_path)
    for w in paradigm_path.read_text().split():
        if w not in stop_words:
            words_in_test_suite.add(w)

print()
print(words_in_test_suite)

num_overlapping = 0
num_total = 0
for w in words_in_test_suite:
    if w in vocab_base:
        num_overlapping += 1
    else:
        print(f'{w:<24} not in Robert-base vocab')
    num_total += 1

ratio = num_overlapping / num_total
print()
print(f'{num_overlapping}/{num_total}={ratio:.2f} of words in test suite in Roberta-base vocab')

# if names are not capitalized, names are split by roberta-base:
print(tokenizer_base.tokenize('This is philip and edward.'))
print(tokenizer_base.tokenize('This is Philip and Edward.'))
