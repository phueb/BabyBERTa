from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    corpora = data / 'corpora'
    tokenizers = data / 'tokenizers'

    # probing data can be found at https://github.com/phueb/Zorro/tree/master/sentences
    probing_sentences = Path('/') / 'media' / 'ludwig_data' / 'Zorro' / 'sentences'
    probing_results = Path.home() / 'Zorro' / 'runs'

    # wikipedia sentences file was created using https://github.com/akb89/witokit
    wikipedia_sentences = Path.home() / 'witokit_download' / 'processed.txt'


class Data:
    min_sentence_length = 3
    train_prob = 1.0  # probability that sentence is assigned to train split
    mask_symbol = '<mask>'
    pad_symbol = '<pad>'
    unk_symbol = '<unk>'
    bos_symbol = '<s>'
    eos_symbol = '</s>'
    roberta_symbols = [mask_symbol, pad_symbol, unk_symbol, bos_symbol, eos_symbol]


class Training:
    feedback_interval = 1000


class Eval:
    interval = 20_000
