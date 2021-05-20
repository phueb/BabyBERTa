from dataclasses import dataclass
from typing import Tuple

param2requests = {
    'corpora': [
        # ('wikipedia3', 'aonewsela', 'aochildes'),
        # ('aochildes', 'aonewsela', 'wikipedia3'),
        # ('wikipedia1', 'wikipedia2', 'wikipedia3'),
        # ('aochildes', 'aonewsela'),
        # ('wikipedia1', 'wikipedia2'),
        ('aochildes',),
        ('aonewsela',),
        ('wikipedia3',),
    ],

    'num_mask_patterns': [6],
    'consecutive_masking': [False],
    'leave_unmasked_prob_start': [0.0],

}

# check
if 'leave_unmasked_prob_start' in param2requests:
    if 'num_epochs' in param2requests:
        for num_epochs in param2requests['num_epochs']:
            if num_epochs != 1:
                # the curriculum for leave_unmasked_prob is WITHIN each epoch
                raise ValueError(f'Using more than one epoch is not compatible with leave_unmasked_prob curriculum.')


param2debug = {
    'num_mask_patterns': 1,
    'num_layers': 2,
    'corpora': ('aochildes',),
}

param2default = {
    # data
    'sample_with_replacement': False,  # this must be False if corpus order is to be preserved during training
    'consecutive_masking': False,  # better dev pp and grammatical accuracy when false
    'num_sentences_per_input': 1,  # if too large -> may exceed CUDA memory, 1 is best for good number-agreement
    'include_punctuation': True,
    'allow_truncated_sentences': False,
    'num_mask_patterns': 3,
    'mask_pattern_size': 2,  # used only if probabilistic_masking = False
    'probabilistic_masking': True,
    'mask_probability': 0.15,  # used only if probabilistic_masking = true
    'leave_unmasked_prob_start': 0.0,  # better performance if no unmasking
    'leave_unmasked_prob': 0.1,
    'random_token_prob': 0.1,
    'corpora': ('aochildes', 'aonewsela', 'wikipedia3'),
    'tokenizer': 'a-a-w-w-w-8192',  # larger than 8k slightly reduces performance
    'add_prefix_space': True,  # better if True, whether to treat first token like any other token (False in GPT-2)
    'max_num_tokens_in_sequence': 128,  # unacceptable performance if lower than ~32

    # training
    'batch_size': 16,
    'lr': 1e-4,  # 1e-4 is used in fairseq (and performs better here), and 1e-3 is default in huggingface
    'num_epochs': 1,
    'num_warmup_steps': 10_000,  # slightly better than 0
    'weight_decay': 0.0,

    # eval
    'score_with_mask': False,  # True to use pseudo-log-likelihoods when probing with forced-choice task, BLIMP style

    # model
    'hidden_size': 256,
    'num_layers': 8,
    'num_attention_heads': 8,
    'intermediate_size': 1024,
    'initializer_range': 0.02,  # stdev of trunc normal for initializing all weights
    'layer_norm_eps': 1e-5,  # 1e-5 default in fairseq (and slightly better performance), 1e-12 default in hgugingface,
}


@dataclass
class Params:
    """
    this object is loaded at the start of job.main() by calling Params.from_param2val(),
    and is populated by Ludwig with hyper-parameters corresponding to a single job.
    """

    # data
    sample_with_replacement: bool
    consecutive_masking: bool
    num_sentences_per_input: int
    include_punctuation: bool
    allow_truncated_sentences: bool
    num_mask_patterns: int
    mask_pattern_size: int
    probabilistic_masking: bool
    mask_probability: float
    leave_unmasked_prob_start: float
    leave_unmasked_prob: float
    random_token_prob: float
    corpora: Tuple[str]
    tokenizer: str
    add_prefix_space: bool
    max_num_tokens_in_sequence: int

    # training
    batch_size: int
    lr: float
    num_epochs: int
    num_warmup_steps: int
    weight_decay: float

    # eval
    score_with_mask: bool

    # model
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    initializer_range: float
    layer_norm_eps: float

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'project_path', 'save_path']}
        return cls(**kwargs)
