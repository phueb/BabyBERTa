import torch
from torch.nn import CrossEntropyLoss
from fairseq import utils
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.gpt2_bpe_utils import Encoder

from babybert import configs
from babybert.io import save_yaml_file
from babybert.io import load_sentences_from_file, save_forced_choice_predictions, save_open_ended_predictions


ROBERTA_NAME = 'roberta-march11'    # 'roberta.base'
CHECKPOINT_NAME = 'checkpoint_last'

NUM_VOCAB = 8192  # used to check the length of the loaded vocab, pytorch hub models may load an unwanted vocab

FORCED_CHOICE = True
OPEN_ENDED = False


def probe_pretrained_roberta(model: RobertaHubInterface,
                             train_step: str,
                             architecture_name: str,
                             ):
    print(f'Num parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    model.eval()

    # determine rep_name  # todo test
    rep = 0
    save_path = configs.Dirs.probing_results / architecture_name / str(rep) / 'saves'
    while save_path.exists():
        rep += 1
        save_path = save_path.parent.parent / str(rep) / 'saves'

    # for each probing task
    assert configs.Dirs.probing_sentences.exists()
    for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):

        task_name = sentences_path.stem
        task_type = sentences_path.parent.name

        # load probing sentences
        print(f'Starting probing with task={task_name}', flush=True)
        probing_sentences = load_sentences_from_file(sentences_path)

        # prepare out path
        probing_results_path = save_path / task_type / f'probing_{task_name}_results_{train_step}.txt'
        if not probing_results_path.parent.exists():
            probing_results_path.parent.mkdir(exist_ok=True, parents=True)

        # save param2val
        param_path = save_path.parent.parent
        if not param_path.is_dir():
            param_path.mkdir(parents=True, exist_ok=True)
        if not (param_path / 'param2val.yaml').exists():
            save_yaml_file(param2val_path=param_path / 'param2val.yaml', architecture=param_path.name)

        if OPEN_ENDED and task_type == 'open_ended':
            predicted_words = []
            for n, sentence in enumerate(probing_sentences):
                with torch.no_grad():
                    for res in model.fill_mask(sentence, topk=3):
                        sentence_out, _, pw = res
                        if pw:  # sometimes empty string is predicted
                            break
                    predicted_words.append(pw)
                    print(f'{n + 1:>6}/{len(probing_sentences):>6} | {make_pretty(sentence_out)}')

            save_open_ended_predictions(probing_sentences, predicted_words, probing_results_path,
                                        verbose=True if 'dummy' in task_name else False)

        if FORCED_CHOICE and task_type == 'forced_choice':
            cross_entropies = []
            loss_fct = CrossEntropyLoss(reduction='none')

            for n, sentence in enumerate(probing_sentences):
                with torch.no_grad():
                    tokens = model.encode(sentence)  # todo batch encode using encode()
                    if tokens.dim() == 1:
                        tokens = tokens.unsqueeze(0)
                    with utils.model_eval(model.model):
                        features, extra = model.model(
                            tokens.long().to(device=model.device),
                            features_only=False,
                            return_all_hiddens=False,
                        )
                    logits_3d = features  # [batch size, seq length, vocab size]
                    # logits need to be [batch size, vocab size, seq length]
                    # tags need to be [batch size, seq length]
                    labels = tokens
                    ce = loss_fct(logits_3d.permute(0, 2, 1), labels).cpu().numpy().mean()
                    cross_entropies.append(ce)

                    print(
                        f'{n + 1:>6}/{len(probing_sentences):>6} | {ce:.2f} {make_pretty(sentence)}')

            save_forced_choice_predictions(probing_sentences, cross_entropies, probing_results_path)


def make_pretty(sentence: str):
    return " ".join([f"{w:<18}" for w in sentence.split()])


if __name__ == '__main__':

    models = []
    steps = []
    names = []

    # load from torch.hub
    try:
        roberta = torch.hub.load('pytorch/fairseq', ROBERTA_NAME)
        print(f'Loaded {ROBERTA_NAME} from torch.hub')
        steps = ['unknown']
        models = [roberta]
        names = [ROBERTA_NAME]

    # load custom models
    except RuntimeError as e:
        print(e)
        for architecture_path in (configs.Dirs.root / 'pretrained_models').glob(ROBERTA_NAME):
            print(f'Loading model from {architecture_path}')
            model_name_or_path = str(architecture_path / 'checkpoints')
            roberta = RobertaModel.from_pretrained(model_name_or_path,
                                                   checkpoint_file=f'{CHECKPOINT_NAME}.pt',
                                                   data_name_or_path=str(architecture_path / 'data-bin'),
                                                   return_dict=False,  # compatibility with transformers v3
                                                   )
            step = CHECKPOINT_NAME.split('_')[-1]
            models.append(roberta)
            steps.append(step)
            names.append(architecture_path.name)

    if not models:
        raise RuntimeError('Did not find models.')

    for model, step, name in zip(models, steps, names):

        # check encoder of model
        """
        to guarantee the correct vocab is loaded, modify cfg in fairseq.fairse.checkpoint_utils by adding:
        
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

        encoder: Encoder = model.bpe.bpe
        vocab = encoder.encoder
        print(f'Found {len(vocab)} words in vocab')
        if not len(vocab) == NUM_VOCAB:
            raise RuntimeError(f'Pretrained model state dict points to vocab that is not of size={NUM_VOCAB}')

        # probe
        probe_pretrained_roberta(model, step, name)





