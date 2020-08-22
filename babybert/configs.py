from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    data = root / 'data'
    probing_sentences = Path('/') / 'media' / 'research_data' / 'Babeval' / 'sentences'
    probing_results = Path.home() / 'Babeval_phueb' / 'runs'
    # probing data can be found at https://github.com/phueb/Babeval/tree/master/sentences


class Data:
    uncased = True  # make sure the correct Google vocab is loaded, e.g. bert-base-uncased-vocab.txt
    min_seq_length = 3
    max_seq_length = 128  # before word-piecing
    train_prob = 0.8  # probability that utterance is assigned to train split
    special_symbols = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    childes_symbols = ['[NAME]', '[PLACE]', '[MISC]']


class Training:
    feedback_interval = 1000


class Eval:
    interval = 10_000
    eval_pp_at_step_zero = False
    batch_size = 512

    probing_names = [
        'dummy',
        'agreement_across_1_adjective',
        'agreement_across_2_adjectives',
        'agreement_between_neighbors',
        'agreement_across_PP',
        'agreement_across_RC',
        'agreement_in_1_verb_question',
        'agreement_in_2_verb_question',
    ]
