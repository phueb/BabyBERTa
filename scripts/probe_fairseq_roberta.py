"""
Probe roberta models trained in fairseq in the same way that BabyBert is probed.

To guarantee the correct vocab is loaded, modify cfg in fairseq.fairseq.checkpoint_utils by adding:

from fairseq.dataclass.utils import overwrite_args_by_name
new_bpe_cfg = {
'_name': 'gpt2',
'gpt2_encoder_json': '/home/ph/BabyBERT/pretrained_models/roberta-feb25/checkpoints/vocab.json',
'gpt2_vocab_bpe': '/home/ph/BabyBERT/pretrained_models/roberta-feb25/checkpoints/merges.txt',
}
overrides = {'bpe': new_bpe_cfg}
overwrite_args_by_name(cfg, overrides)
print(cfg['bpe'])
"""
import shutil
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.gpt2_bpe_utils import Encoder
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

from babybert import configs
from babybert.probing import do_probing
from babybert.io import save_yaml_file
from babybert.params import Params, param2default

MODEL_DATA_FOLDER_NAME = 'fairseq_custom_march11'

NUM_VOCAB = 8192  # used to check the length of the loaded vocab, pytorch hub models may load an unwanted vocab


if __name__ == '__main__':

    assert configs.Dirs.probing_sentences.exists()

    framework, implementation, configuration = MODEL_DATA_FOLDER_NAME.split('_')

    # remove previous results
    path_model_results = configs.Dirs.probing_results / MODEL_DATA_FOLDER_NAME
    if path_model_results.exists():
        shutil.rmtree(path_model_results)

    path_model_data = configs.Dirs.root / 'fairseq_models' / MODEL_DATA_FOLDER_NAME

    for path_checkpoint in path_model_data.rglob('checkpoint_last.pt'):

        rep = path_checkpoint.parent.name

        # load model
        print(f'Loading model from {path_checkpoint}')
        model = RobertaModel.from_pretrained(model_name_or_path=str(path_model_data / rep),
                                             checkpoint_file=str(path_checkpoint),
                                             data_name_or_path=str(path_model_data / 'data-bin'),
                                             )
        print(f'Num parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
        model.eval()

        # get step
        state = load_checkpoint_to_cpu(str(path_checkpoint))
        step = state['cfg']['model'].total_num_update

        print(state['cfg']['model'])
        # raise SystemExit

        # check that configuration really corresponds to reference configuration
        if configuration == 'reference':
            params = Params.from_param2val(param2default)
            assert state['cfg']['model'].max_sentences == params.batch_size
            assert state['cfg']['model'].random_token_prob == 0.1
            assert state['cfg']['model'].leave_unmasked_prob == 0.1
            assert state['cfg']['model'].mask_prob == params.mask_probability
            assert state['cfg']['model'].sample_break_mode == 'eos'
            assert state['cfg']['model'].encoder_attention_heads == params.num_attention_heads
            assert state['cfg']['model'].encoder_embed_dim == params.hidden_size
            assert state['cfg']['model'].encoder_ffn_embed_dim == params.intermediate_size
            assert state['cfg']['model'].encoder_layers == params.num_layers
            assert state['cfg']['model'].lr == [params.lr]

        # check encoder of model
        encoder: Encoder = model.bpe.bpe
        vocab = encoder.encoder
        print(f'Found {len(vocab)} words in vocab')
        if not len(vocab) == NUM_VOCAB:
            raise RuntimeError(f'Pretrained model state dict points to vocab that is not of size={NUM_VOCAB}')

        # make new save_path
        save_path = path_model_results / str(rep) / 'saves'
        if not save_path.is_dir():
            save_path.mkdir(parents=True, exist_ok=True)

        # save basic model info
        if not (path_model_results / 'param2val.yaml').exists():
            save_yaml_file(path_out=path_model_results / 'param2val.yaml',
                           param2val={'framework': framework,
                                      'is_official': True if implementation == 'official' else False,
                                      'is_reference': True if configuration == 'reference' else False,
                                      'is_base': False,
                                      })

        # for each probing task
        for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):
            do_probing(save_path,
                       sentences_path,
                       model,
                       step,
                       include_punctuation=True,
                       score_with_mask=False)





