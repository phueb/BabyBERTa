import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from fairseq.models.roberta import RobertaModel

from babybert import configs
from babybert.io import save_yaml_file
from babybert.io import load_utterances_from_file, save_forced_choice_predictions, save_open_ended_predictions


INCLUDE_PUNCTUATION = True

if __name__ == '__main__':

    for architecture_path in (configs.Dirs.root / 'pretrained_models').glob('*'):

        roberta = RobertaModel.from_pretrained(str(architecture_path / 'checkpoints'),
                                               checkpoint_file='checkpoint_best.pt',
                                               data_name_or_path=str(architecture_path / 'data-bin' / 'childes'),
                                               )

        # roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'data-bin/childes')
        # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')


        print(f'Num parameters={sum(p.numel() for p in roberta.parameters() if p.requires_grad):,}')
        roberta.eval()

        step = 'best'
        rep_name = '0'
        save_path = configs.Dirs.probing_results / architecture_path.name / rep_name / 'saves'

        assert configs.Dirs.probing_sentences.exists()

        # for each probing task
        for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):

            task_name = sentences_path.stem
            task_type = sentences_path.parent.name

            # load probing sentences
            print(f'Starting probing with task={task_name}', flush=True)
            sentences_in = load_utterances_from_file(sentences_path, include_punctuation=INCLUDE_PUNCTUATION)

            # TODO test
            sentences_in = [
                'one two three <mask>'.split(),
                'one two <mask> four'.split(),
                'my name is <mask>'.split(),
                '<mask> book is on the table'.split(),
            ]

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

                        # TODO remove
                        print(roberta.fill_mask(sentence_str, topk=3))

                        res = roberta.fill_mask(sentence_str.replace(configs.Data.mask_symbol, '<mask>'), topk=1)
                        sentence_out = res[0][0].split()
                        sentences_out.append(sentence_out)
                        sentence_out_string = ' '.join([f'{w:>12}' for w in sentence_out])
                        print(f'{n:>6}/{len(sentences_in):>6} | {sentence_out_string}')

                raise SystemExit
                save_open_ended_predictions(sentences_in, sentences_out, probing_results_path,
                                            verbose=True if 'dummy' in task_name else False)

            elif task_type == 'forced_choice':
                continue

                cross_entropies = []

                for sentence in sentences_in:
                    with torch.no_grad():
                        sentence_str = ' '.join(sentence)

                save_forced_choice_predictions(sentences_in, cross_entropies, probing_results_path)
            else:
                raise AttributeError('Invalid arg to "task_type".')





