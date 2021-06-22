"""
Train a ByteLevel-BPE on custom corpora
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing, RobertaProcessing

from babyberta import configs


VOCAB_SIZE = 4096 * 2
MIN_FREQUENCY = 2
CORPUS_NAMES = ['aochildes', 'aonewsela', 'wikipedia1', 'wikipedia2', 'wikipedia3']
ADD_PREFIX_SPACE = True

model = BPE(unk_token=configs.Data.unk_symbol)
tokenizer = Tokenizer(model)
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=ADD_PREFIX_SPACE)
tokenizer.normalizer = Lowercase()

corpus_file_paths = [str(configs.Dirs.corpora / f'{name}.txt') for name in CORPUS_NAMES]
trainer = BpeTrainer(special_tokens=configs.Data.roberta_symbols,
                     vocab_size=VOCAB_SIZE,
                     min_frequency=MIN_FREQUENCY,
                     )
tokenizer.train(corpus_file_paths, trainer)

# add additional info
tokenizer.post_processor = RobertaProcessing(sep=('</s>', tokenizer.token_to_id("</s>")),
                                             cls=('<s>', tokenizer.token_to_id("<s>")),
                                             add_prefix_space=ADD_PREFIX_SPACE)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id(configs.Data.pad_symbol), pad_token=configs.Data.pad_symbol)
tokenizer.enable_truncation(max_length=128)

# save tokenizer
json_path = configs.Dirs.tokenizers / 'custom_tokenizer.json'
tokenizer.save(str(json_path), pretty=True)

print(f'Saved tokenizer config to {json_path}')
