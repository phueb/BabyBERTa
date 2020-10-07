
param2requests = {
    # 'corpus_name': ['childes-20191206', 'newsela'],
}


param2debug = {
    'num_masked': 1,
    'num_layers': 2,
    'num_utterances_per_input': 1,
}

param2default = {
    # data
    'num_utterances_per_input': 1,  # if too large -> may exceed CUDA memory, must be 1 to get good number-agreement
    'include_punctuation': True,
    'training_order': 'age-ordered',
    'num_masked': 6,
    'corpus_name': 'childes-20191206',
    'bbpe': 'c-16384',

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
