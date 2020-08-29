from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    probing_sentences = Path('/') / 'media' / 'research_data' / 'Babeval' / 'sentences'
    probing_results = Path.home() / 'Babeval_phueb' / 'runs'
    # probing data can be found at https://github.com/phueb/Babeval/tree/master/sentences


class Data:
    uncased = True  # make sure the correct Google vocab is loaded, e.g. bert-base-uncased-vocab.txt
    min_utterance_length = 3
    max_utterance_length = 32  # before word-piecing and concatenation of utterances
    max_word_length = 20  # reduces amount of word-pieces for long words
    train_prob = 0.8  # probability that utterance is assigned to train split
    special_symbols = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    childes_symbols = ['[NAME]', '[PLACE]', '[MISC]'] + ['[LONG]']


class Training:
    feedback_interval = 1000


class Eval:
    interval = 20_000
    eval_pp_at_step_zero = False
    batch_size = 512

