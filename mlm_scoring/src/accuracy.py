""""
Collect pseudo-log-likelihoods saved in output folder, and output summary table.

Odd-numbered lines contain grammatical sentences, an even-numbered lines contain un-grammatical sentences.
"""
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from collections import defaultdict

from babyberta import configs


def calc_and_print_accuracy(data_name: str,  # 'zorro' or 'blimp
                            lower_case: bool,
                            ):

    # get paradigm2phenomenon
    if data_name == 'blimp':
        from mlm_scoring.blimp.helper import file_name2phenomenon
        num_expected_pairs = 1000
    elif data_name == 'zorro':
        from mlm_scoring.zorro.helper import file_name2phenomenon
        num_expected_pairs = 2000
    else:
        raise AttributeError('Invalid arg to "data_name".')
    phenomena = set([v for v in file_name2phenomenon.values()])
    phenomenon2fns = {phenomenon: [pa for pa, ph in file_name2phenomenon.items() if ph == phenomenon]
                      for phenomenon in phenomena}

    path_to_output_dir = Path(data_name) / 'output' / f'lower_case={lower_case}'
    phenomenon2col = defaultdict(list)
    for model_dir in path_to_output_dir.glob('*'):

        model_name = model_dir.name
        print('********************************')
        print(model_name)
        print('********************************')

        phenomenon2col['Model'].append(model_name.replace('_', '+'))

        base_dir = configs.Dirs.mlm_scoring / data_name / 'output' / f'lower_case={lower_case}' / model_name

        assert base_dir.exists()

        for phenomenon, fns in phenomenon2fns.items():

            accuracies = []
            for fn in fns:
                file = base_dir / f'{fn}.txt'
                with file.open('rt') as f:
                    scores = f.readlines()
                num_pairs = len(scores) // 2

                # compute accuracy for paradigm
                count = 0
                for i in range(num_pairs):
                    if float(scores[2*i]) > float(scores[2*i+1]):
                        count += 1
                if len(scores) == 0:
                    logging.error("{} is empty, skipping".format(file))
                    continue
                assert num_pairs == num_expected_pairs
                acc = count / num_pairs
                accuracies.append(acc * 100)

            # collect
            phenomenon_rotated = '\rot{' + phenomenon.capitalize() + '}'  # /rot{} is a custom latex command
            phenomenon2col[phenomenon_rotated].append(np.mean(accuracies))
            # Since all 67 classes have 1000 pairs, per-class and overall accuracies are the desired (micro)averages

    df = pd.DataFrame(data=phenomenon2col)
    df = df[['Model'] + [n for n in sorted(df.columns) if n != 'Model']]
    print(df.round(1).to_latex(index=False, bold_rows=True, escape=False))

    print()
    print(f'lower_case={lower_case}')
    df['Overall'] = df.mean(axis=1)
    df = df.sort_values(axis=0, by='Overall')
    for model_name, overall_acc in zip(df['Model'], df['Overall'].round(2)):
        print(f'{model_name:<40} {overall_acc}')
