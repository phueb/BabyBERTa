import torch
from transformers.models.roberta import RobertaForMaskedLM, RobertaConfig

from mlm.scorers import MLMScorerPT

import mxnet as mx
ctxs = [mx.gpu()]


from babyberta.params import Params, param2default
from babyberta.utils import load_tokenizer
from babyberta import configs

params = Params.from_param2val(param2default)

# Byte-level BPE tokenizer
path_tokenizer_config = configs.Dirs.tokenizers / f'{params.tokenizer}.json'
tokenizer = load_tokenizer(path_tokenizer_config, params.max_num_tokens_in_sequence)
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)

# load model with config
config = RobertaConfig(vocab_size=vocab_size,
                       pad_token_id=tokenizer.token_to_id(configs.Data.pad_symbol),
                       bos_token_id=tokenizer.token_to_id(configs.Data.bos_symbol),
                       eos_token_id=tokenizer.token_to_id(configs.Data.eos_symbol),
                       return_dict=True,
                       is_decoder=False,
                       is_encoder_decoder=False,
                       add_cross_attention=False,
                       layer_norm_eps=params.layer_norm_eps,  # 1e-5 used in fairseq
                       max_position_embeddings=params.max_num_tokens_in_sequence + 2,
                       hidden_size=params.hidden_size,
                       num_hidden_layers=params.num_layers,
                       num_attention_heads=params.num_attention_heads,
                       intermediate_size=params.intermediate_size,
                       initializer_range=params.initializer_range,
                       )
model = RobertaForMaskedLM(config=config)
model.cuda(0)

# load model weights using pytorch state_dict
path_ckpt = configs.Dirs.root / 'saved_models' / 'model.pt'  # this model was trained on AO-CHILDES
print(f'Trying to load model from {path_ckpt}')
state_dict = torch.load(path_ckpt)
model.load_state_dict(state_dict)
print(f'Loaded model from {path_ckpt}')
print('Number of parameters: {:,}'.format(model.num_parameters()), flush=True)

vocab = None
scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)
print(scorer.score_sentences(["Hello world!"]))
