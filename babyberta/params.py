import attr

param2requests = {
    'corpus_name': ['wiki-20191017-hebb-3M_tokenized', 'childes-20201026'],

    'leave_unmasked_prob': [0.1, 0.0],


}


param2debug = {
    'num_mask_patterns': 1,
    'num_layers': 2,
}

param2default = {
    # data
    'sample_with_replacement': False,
    'consecutive_masking': False,  # better dev pp when false
    'num_sentences_per_input': 1,  # if too large -> may exceed CUDA memory, 1 is best for good number-agreement
    'include_punctuation': True,
    'allow_truncated_sentences': False,
    'training_order': 'none',  # 'age-ordered' is better for CHILDES data - must use with consecutive_masking=True
    'num_mask_patterns': 6,  # 6 is better than lower or
    'mask_pattern_size': 2,  # used only if probabilistic_masking = False
    'probabilistic_masking': True,
    'mask_probability': 0.15,  # used only if probabilistic_masking = true
    'leave_unmasked_prob': 0.0,  # setting this to zero makes BabyBERTa perform better than standard Roberta
    'random_token_prob': 0.1,
    'corpus_name': 'childes-20201026',
    'bbpe': 'c-n-w-8192',  # larger than 8k slightly reduces performance
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


@attr.s
class Params(object):
    # data
    sample_with_replacement = attr.ib(validator=attr.validators.instance_of(bool))
    consecutive_masking = attr.ib(validator=attr.validators.instance_of(bool))
    num_sentences_per_input = attr.ib(validator=attr.validators.instance_of(int))
    training_order = attr.ib(validator=attr.validators.instance_of(str))
    include_punctuation = attr.ib(validator=attr.validators.instance_of(bool))
    allow_truncated_sentences = attr.ib(validator=attr.validators.instance_of(bool))
    num_mask_patterns = attr.ib(validator=attr.validators.instance_of(int))
    mask_pattern_size = attr.ib(validator=attr.validators.instance_of(int))
    probabilistic_masking = attr.ib(validator=attr.validators.instance_of(bool))
    mask_probability = attr.ib()
    leave_unmasked_prob = attr.ib(validator=attr.validators.instance_of(float))
    random_token_prob = attr.ib(validator=attr.validators.instance_of(float))
    corpus_name = attr.ib(validator=attr.validators.instance_of(str))
    bbpe = attr.ib(validator=attr.validators.instance_of(str))
    add_prefix_space = attr.ib(validator=attr.validators.instance_of(bool))
    max_num_tokens_in_sequence = attr.ib(validator=attr.validators.instance_of(int))

    # training
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    num_epochs = attr.ib(validator=attr.validators.instance_of(int))
    num_warmup_steps = attr.ib(validator=attr.validators.instance_of(int))
    weight_decay = attr.ib(validator=attr.validators.instance_of(float))

    # eval
    score_with_mask = attr.ib(validator=attr.validators.instance_of(bool))

    # model
    num_layers = attr.ib(validator=attr.validators.instance_of(int))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    num_attention_heads = attr.ib(validator=attr.validators.instance_of(int))
    intermediate_size = attr.ib(validator=attr.validators.instance_of(int))
    initializer_range = attr.ib(validator=attr.validators.instance_of(float))
    layer_norm_eps = attr.ib(validator=attr.validators.instance_of(float))

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