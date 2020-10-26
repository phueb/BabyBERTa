"""
Train a B-BPE on custom corpora
"""
from tokenizers import ByteLevelBPETokenizer


from babybert import configs


VOCAB_SIZE = 4096 * 8  # TODO what's good here?
MIN_FREQUENCY = 2
CORPUS_NAMES = ['childes-20191206']

tokenizer = ByteLevelBPETokenizer(lowercase=False)
corpus_file_paths = [str(configs.Dirs.corpora / f'{name}.txt') for name in CORPUS_NAMES]
special_tokens = configs.Data.universal_symbols + configs.Data.roberta_symbols + configs.Data.childes_symbols
tokenizer.train(files=corpus_file_paths,
                vocab_size=VOCAB_SIZE,
                min_frequency=MIN_FREQUENCY,
                special_tokens=special_tokens,
                )

# save tokenizer
name = '-'.join([n[0] for n in CORPUS_NAMES]) + '-' + str(VOCAB_SIZE)
bbpe_path = configs.Dirs.tokenizers / name
if not bbpe_path.exists():
    bbpe_path.mkdir(exist_ok=True, parents=True)
tokenizer.save_model(str(bbpe_path))

print(tokenizer)