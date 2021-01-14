from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    corpora = data / 'corpora'
    tokenizers = data / 'tokenizers'
    probing_sentences = Path('/') / 'media' / 'research_data' / 'Zorro' / 'sentences'
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
    add_prefix_space = True

    max_num_tokens_in_sequence = 256


class Training:
    feedback_interval = 1000


class Eval:
    interval = 20_000
    eval_pp = False
    batch_size = 128  # 128 is too large when vocab size ~ 8k

