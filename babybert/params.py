
param2requests = {
    'corpus_name': ['wiki-20191017-hebb-3M_tokenized', 'childes-20201026', 'newsela'],
    # 'corpus_name': ['childes-20201026'],

    'weight_decay': [0.0, 0.1],
    # 'hidden_size': [768],
    # 'num_layers': [12],
    # 'num_attention_heads': [12],
    # 'intermediate_size': [1024],
}


param2debug = {
    'num_mask_patterns': 1,
    'mask_pattern_size': 1,
    'num_layers': 2,
    'num_sentences_per_input': 1,
}

param2default = {
    # data
    'consecutive_masking': False,  # better dev pp when false
    'num_sentences_per_input': 1,  # if too large -> may exceed CUDA memory, 1 is best for good number-agreement
    'include_punctuation': True,
    'allow_truncated_sentences': False,
    'training_order': 'none',  # 'age-ordered' is better for CHILDES data - must use with consecutive_masking=True
    'num_mask_patterns': 6, # 6 is better than lower or higher
    'mask_pattern_size': 2,  # 2 is better than 1 and as good as 3
    'leave_unmasked_prob': 0.0,
    'random_token_prob': 0.0,
    'corpus_name': 'childes-20201026',
    'bbpe': 'c-n-w-8192',  # larger than 8k slightly reduces performance
    'add_prefix_space': True,  # better if True, whether to treat first token like any other token (False in GPT-2)
    'max_num_tokens_in_sequence': 128,  # unacceptable performance if lower than ~32

    # training
    'batch_size': 16,
    'lr': 1e-4,
    'num_epochs': 1,
    'num_warmup_steps': 10_000,  # slightly better than 0
    'weight_decay': 0.0,

    # model
    'hidden_size': 256,
    'num_layers': 8,
    'num_attention_heads': 8,
    'intermediate_size': 1024,
}
