
param2requests = {
    'training_order': ['age-ordered', 'age-reversed'],  # TODO
    'include_punctuation': [False],  # TODO test
}


param2debug = {
    'num_masked': 1,
    'num_layers': 2,
    'google_vocab_rule': 'none',
}

param2default = {
    'include_punctuation': True,
    'batch_size': 16,
    'lr': 1e-4,
    'training_order': 'none',
    'hidden_size': 256,
    'num_layers': 8,
    'num_attention_heads': 8,
    'intermediate_size': 1024,
    'num_epochs': 1,
    'num_masked': 6,
    'corpus_name': 'childes-20191206',
    'childes_vocab_size': 4000,
    'google_vocab_rule': 'inclusive',
}
