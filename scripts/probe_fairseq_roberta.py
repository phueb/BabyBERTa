"""
Probe roberta models trained in fairseq in the same way that BabyBERTa is probed.

Test sentences are lower-cased because Roberta-base is case sensitive,
because we trained Roberta-base on lower-cased data.

We found better performance on grammatical accuracy when training Roberta-base with smaller batch size, 256.

The folders named integers represent a different random initialisation (or "replication").
Because of idiosyncrasies of loading the model data in fairseq,
 the folder containing model data are inside the "replication" folder.

"""
import shutil
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.gpt2_bpe_utils import Encoder
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

from babyberta import configs
from babyberta.probing import do_probing
from babyberta.io import save_yaml_file


# TODO it would be faster to convert fairseq model to huggingface model before evaluation


MODEL_DATA_FOLDER_NAMES = [
    'fairseq_RoBERTa-base_AO-CHILDES',
    'fairseq_RoBERTa-base_Wikipedia-1',
]
LAST_OR_BEST = 'last'  # last is better than best usually


if __name__ == '__main__':

    assert configs.Dirs.probing_sentences.exists()

    # for all reps
    for path_rep in (configs.Dirs.root / 'fairseq_models').glob('*'):

        rep = path_rep.name
        print(f'rep={rep}')

        # for all model groups
        for path_model_data in path_rep.glob('*'):

            model_data_folder_name = path_model_data.name

            if model_data_folder_name not in MODEL_DATA_FOLDER_NAMES:
                continue

            path_model_results = configs.Dirs.probing_results / model_data_folder_name

            # remove previous results
            path_remove = path_model_results / str(rep)
            if path_remove.exists():
                shutil.rmtree(path_remove)

            framework, architecture, corpora = model_data_folder_name.split('_')

            path_checkpoint = path_model_data / f'checkpoint_{LAST_OR_BEST}.pt'

            # load model
            print(f'Loading model from {path_model_data}')
            print(f'Loading checkpoint {path_checkpoint}')
            if 'AO-CHILDES' in corpora:
                bin_name = 'aochildes-data-bin'
                data_size = '5M'
            elif 'Wikipedia-1' in corpora:
                bin_name = 'wikipedia1_new1_seg'
                data_size = '13M'
            else:
                raise AttributeError('Invalid data size for fairseq model.')
            model = RobertaModel.from_pretrained(model_name_or_path=str(path_model_data),
                                                 checkpoint_file=str(path_checkpoint),
                                                 data_name_or_path=str(path_model_data / bin_name),
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
                                          'corpora': corpora,
                                          'data_size': data_size,
                                          })

            # for each paradigm
            for paradigm_path in configs.Dirs.probing_sentences.rglob('*.txt'):
                do_probing(save_path,
                           paradigm_path,
                           model,
                           step,
                           include_punctuation=True,
                           lower_case=True,
                           )





