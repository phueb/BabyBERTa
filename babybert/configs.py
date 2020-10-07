from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    corpora = data / 'corpora'
    tokenizers = data / 'tokenizers'
    probing_sentences = Path('/') / 'media' / 'research_data' / 'Babeval' / 'sentences'
    probing_results = Path.home() / 'Babeval_phueb' / 'runs'
    # probing data can be found at https://github.com/phueb/Babeval/tree/master/sentences


class Data:
    lowercase_input = True
    min_utterance_length = 3
    max_utterance_length = 30  # must work for all corpora. before sub-tokenization and concatenation of utterances
    max_word_length = 20  # reduces amount of sub-tokens for long words
    train_prob = 0.8  # probability that utterance is assigned to train split
    long_symbol = '[LONG]'  # this is used in training sentences regardless of the tokenizer of the model
    mask_symbol = '[MASK]'  # this is used in probing sentences regardless of the tokenizer of the model
    universal_symbols = [mask_symbol, long_symbol]
    roberta_symbols = ['<pad>', '<unk>', '<s>', '</s>']
    childes_symbols = ['[NAME]', '[PLACE]', '[MISC]']


class Training:
    feedback_interval = 1000


class Eval:
    interval = 20_000
    eval_pp_at_step_zero = False
    batch_size = 256  # 512 is too large for newsela

