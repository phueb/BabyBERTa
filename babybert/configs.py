from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    corpora = data / 'corpora'
    tokenizers = data / 'tokenizers'
    probing_sentences = Path('/') / 'media' / 'ludwig_data' / 'Zorro' / 'sentences'
    probing_results = Path.home() / 'Zorro' / 'runs'
    # probing data can be found at https://github.com/phueb/Zorro/tree/master/sentences


class Data:
    lowercase_input = True
    min_sentence_length = 3
    max_sentence_length = 30  # must work for all corpora. before sub-tokenization and concatenation of sentences
    max_word_length = 20  # words with more characters are replaced by <long> symbol
    train_prob = 1.0  # probability that sentence is assigned to train split
    long_symbol = '<long>'  # this is used in training sentences regardless of the tokenizer of the model
    mask_symbol = '<mask>'
    universal_symbols = [long_symbol]
    roberta_symbols = [mask_symbol, '<pad>', '<unk>', '<s>', '</s>']


class Training:
    feedback_interval = 1000


class Eval:
    interval = 20_000
