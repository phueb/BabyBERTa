"""
Probe roberta models trained in fairseq in the same way that BabyBERTa is probed.

We found better performance on grammatical accuracy when training Roberta-base with smaller batch size, 256.
"""
import shutil
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.gpt2_bpe_utils import Encoder
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

from babyberta import configs
from babyberta.probing import do_probing
from babyberta.io import save_yaml_file


MODEL_DATA_FOLDER_NAME = 'fairseq_Roberta-base_5M'
REP = 2
LAST_OR_BEST = 'last'  # last is better than best


if __name__ == '__main__':

    assert configs.Dirs.probing_sentences.exists()

    framework, architecture, data_size = MODEL_DATA_FOLDER_NAME.split('_')

    # remove previous results
    path_model_results = configs.Dirs.probing_results / MODEL_DATA_FOLDER_NAME
    if path_model_results.exists():
        shutil.rmtree(path_model_results)

    path_model_data = configs.Dirs.root / 'fairseq_models' / MODEL_DATA_FOLDER_NAME

    for path_checkpoint in path_model_data.rglob(f'checkpoint_{LAST_OR_BEST}.pt'):

        rep = path_checkpoint.parent.name

        if int(rep) != int(REP):
            continue

        # load model
        print(f'Loading model from {str(path_model_data / rep)}')
        print(f'Loading checkpoint {str(path_checkpoint)}')
        model = RobertaModel.from_pretrained(model_name_or_path=str(path_model_data / rep),
                                             checkpoint_file=str(path_checkpoint),
                                             data_name_or_path=str(path_model_data / 'aochildes-data-bin'),
                                             )
        print(f'Num parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
        model.eval()

        model.cuda(0)
        print(f'model.device={model.device}')

        # get step
        state = load_checkpoint_to_cpu(str(path_checkpoint))
        step = state['args'].total_num_update

        # check encoder of model
        encoder: Encoder = model.bpe.bpe
        num_vocab = len(encoder.encoder)
        print(f'Found {num_vocab} words in vocab')

        # make new save_path
        save_path = path_model_results / str(rep) / 'saves'
        if not save_path.is_dir():
            save_path.mkdir(parents=True, exist_ok=True)

        # save basic model info
        if not (path_model_results / 'param2val.yaml').exists():
            save_yaml_file(path_out=path_model_results / 'param2val.yaml',
                           param2val={'framework': framework,
                                      'architecture': architecture,
                                      'data_size': data_size,
                                      })

        # for each probing task
        for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):
            do_probing(save_path,
                       sentences_path,
                       model,
                       step,
                       include_punctuation=True,
                       )





