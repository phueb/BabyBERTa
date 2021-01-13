
param2requests = {
    # 'corpus_name': ['wiki-20191017-hebb-3M_tokenized', 'childes-20201026', 'newsela'],
    'corpus_name': ['newsela'],
    'mask_pattern_size': [1, 2, 3, 4],    # todo test
    'num_mask_patterns': [8],

    # 'hidden_size': [768],
    # 'num_layers': [12],
    # 'num_attention_heads': [12],
    # 'intermediate_size': [1024],
}


param2debug = {
    'num_mask_patterns': 1,
    'num_layers': 2,
    'num_sentences_per_input': 1,
}

param2default = {
    # data
    'consecutive_masking': False,  # better dev pp when false
    'num_sentences_per_input': 1,  # if too large -> may exceed CUDA memory, must be 1 to get good number-agreement
    'include_punctuation': True,
    'training_order': 'none',  # 'age-ordered' is better for CHILDES data
    'num_mask_patterns': 8,
    'mask_pattern_size': 1,
    'corpus_name': 'childes-20201026',
    'bbpe': 'c-n-w-8192',

    # training
    'batch_size': 16,
    'lr': 1e-4,
    'num_epochs': 1,
    'num_warmup_steps': 10_000,  # slightly better than 0

    # model
    'hidden_size': 256,
    'num_layers': 8,
    'num_attention_heads': 8,
    'intermediate_size': 1024,
}
