
param2requests = {
    'training_order': ['age-ordered', 'age-reversed']  # TODO
}

# With num_masked=1, made 0,575,465 instances -> 035,966 train MLM batches (when batch-size=16)
# With num_masked=3, made XXXXXXXXX instances -> 107,964 train MLM batches (when batch-size=16)
# With num_masked=6, made 2,976,614 instances -> 186,038 train MLM batches (when batch-size=16)

param2debug = {
    'num_masked': 1,
    'num_layers': 2,
    'google_vocab_rule': 'exclusive',
}

param2default = {
    'batch_size': 16,
    'lr': 1e-4,
    'training_order': 'age-ordered',
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
