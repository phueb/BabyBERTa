import torch
from torch.nn import CrossEntropyLoss
from fairseq import utils
from fairseq.models.roberta import RobertaModel

from babybert import configs
from babybert.io import save_yaml_file
from babybert.io import load_sentences_from_file, save_forced_choice_predictions, save_open_ended_predictions


CHECKPOINT_OR_PRETRAINED_NAME = 'checkpoint_last'  # 'roberta.base'


if __name__ == '__main__':

    for architecture_path in (configs.Dirs.root / 'pretrained_models').glob('*'):

        # load from torch.hub
        try:
            roberta = torch.hub.load('pytorch/fairseq', CHECKPOINT_OR_PRETRAINED_NAME)
        except RuntimeError as e:
            print(e)
            step = CHECKPOINT_OR_PRETRAINED_NAME.split('_')[-1]
        else:
            print(f'Loaded {CHECKPOINT_OR_PRETRAINED_NAME} from torch.hub')
            step = 'best'

        # load custom model
        print(f'Loading model from {architecture_path}')
        roberta = RobertaModel.from_pretrained(str(architecture_path / 'checkpoints'),
                                               checkpoint_file=f'{CHECKPOINT_OR_PRETRAINED_NAME}.pt',
                                               data_name_or_path=str(architecture_path / 'data-bin' / 'childes'),
                                               )

        print(f'Num parameters={sum(p.numel() for p in roberta.parameters() if p.requires_grad):,}')
        roberta.eval()

        rep_name = '0'
        save_path = configs.Dirs.probing_results / architecture_path.name / rep_name / 'saves'

        # for each probing task
        assert configs.Dirs.probing_sentences.exists()
        for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):

            task_name = sentences_path.stem
            task_type = sentences_path.parent.name

            # load probing sentences
            print(f'Starting probing with task={task_name}', flush=True)
            sentences_in = load_sentences_from_file(sentences_path)

            # prepare out path
            probing_results_path = save_path / task_type / f'probing_{task_name}_results_{step}.txt'
            if not probing_results_path.parent.exists():
                probing_results_path.parent.mkdir(exist_ok=True, parents=True)

            # save param2val
            param_path = save_path.parent.parent
            if not param_path.is_dir():
                param_path.mkdir(parents=True, exist_ok=True)
            if not (param_path / 'param2val.yaml').exists():
                save_yaml_file(param2val_path=param_path / 'param2val.yaml', architecture=param_path.name)

            if task_type == 'open_ended':
                sentences_out = []
                for n, sentence in enumerate(sentences_in):
                    with torch.no_grad():
                        sentence_str = ' '.join(sentence)
                        res = roberta.fill_mask(sentence_str, topk=3)
                        sentence_out = res[0][0].split()
                        sentences_out.append(sentence_out)
                        sentence_out_string = ' '.join([f'{w:>12}' for w in sentence_out])
                        print(f'{n+1:>6}/{len(sentences_in):>6} | {sentence_out_string}')

                save_open_ended_predictions(sentences_in, sentences_out, probing_results_path,
                                            verbose=True if 'dummy' in task_name else False)

            elif task_type == 'forced_choice':
                cross_entropies = []
                loss_fct = CrossEntropyLoss(reduction='none')

                for n, sentence in enumerate(sentences_in):
                    with torch.no_grad():
                        sentence_str = ' '.join(sentence)
                        tokens = roberta.encode(sentence_str)
                        if tokens.dim() == 1:
                            tokens = tokens.unsqueeze(0)
                        with utils.model_eval(roberta.model):
                            features, extra = roberta.model(
                                tokens.long().to(device=roberta.device),
                                features_only=False,
                                return_all_hiddens=False,
                            )
                        logits_3d = features  # [batch size, seq length, vocab size]
                        # logits need to be [batch size, vocab size, seq length]
                        # tags need to be [batch size, seq length]
                        labels = tokens
                        ce = loss_fct(logits_3d.permute(0, 2, 1), labels).cpu().numpy().mean()
                        cross_entropies.append(ce)

                        print(f'{n + 1:>6}/{len(sentences_in):>6} | {ce:.2f} {" ".join([f"{w:<16}" for w in sentence_str.split()])}')

                save_forced_choice_predictions(sentences_in, cross_entropies, probing_results_path)
            else:
                raise AttributeError('Invalid arg to "task_type".')





