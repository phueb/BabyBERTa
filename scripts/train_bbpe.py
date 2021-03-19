"""
Train a ByteLevel-BPE on custom corpora
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Lowercase

from babyberta import configs


VOCAB_SIZE = 4096 * 2
MIN_FREQUENCY = 10  # was 10 before march 11, 2021
CORPUS_NAMES = ['childes-20201026', 'newsela', 'wiki-20191017-hebb-3M_tokenized']
ADD_PREFIX_SPACE = True

tokenizer = Tokenizer(BPE(unk_token=configs.Data.unk_symbol))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=ADD_PREFIX_SPACE)
tokenizer.normalizer = Lowercase()

corpus_file_paths = [str(configs.Dirs.corpora / f'{name}.txt') for name in CORPUS_NAMES]
trainer = BpeTrainer(special_tokens=configs.Data.roberta_symbols,
                     vocab_size=VOCAB_SIZE,
                     min_frequency=MIN_FREQUENCY,
                     )
tokenizer.train(corpus_file_paths, trainer)

# save tokenizer
name = '-'.join([n[0] for n in CORPUS_NAMES]) + '-' + str(VOCAB_SIZE)
json_path = configs.Dirs.tokenizers / f'{name}.json'
tokenizer.save(str(json_path))

print(f'Saved tokenizer config to {json_path}')
