""""
Adapted from https://github.com/awslabs/mlm-scoring

Collect pseudo-log-likelihoods collected in output/, and output summary table
"""
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from collections import defaultdict


paradigm2phenomenon = {

    'anaphor_gender_agreement': 'anaphor agreement',
    'anaphor_number_agreement': 'anaphor agreement',

    'animate_subject_passive': 'argument structure',
    'animate_subject_trans': 'argument structure',
    'causative': 'argument structure',
    'drop_argument': 'argument structure',
    'inchoative': 'argument structure',
    'intransitive': 'argument structure',
    'passive_1': 'argument structure',
    'passive_2': 'argument structure',
    'transitive': 'argument structure',

    'principle_A_c_command': 'binding',
    'principle_A_case_1': 'binding',
    'principle_A_case_2': 'binding',
    'principle_A_domain_1': 'binding',
    'principle_A_domain_2': 'binding',
    'principle_A_domain_3': 'binding',
    'principle_A_reconstruction': 'binding',

    'existential_there_object_raising': 'control/raising',
    'existential_there_subject_raising': 'control/raising',
    'expletive_it_object_raising': 'control/raising',
    'tough_vs_raising_1': 'control/raising',
    'tough_vs_raising_2': 'control/raising',

    'determiner_noun_agreement_1': 'determiner-noun agreement',
    'determiner_noun_agreement_2': 'determiner-noun agreement',
    'determiner_noun_agreement_irregular_1': 'determiner-noun agreement',
    'determiner_noun_agreement_irregular_2': 'determiner-noun agreement',
    'determiner_noun_agreement_with_adjective_1': 'determiner-noun agreement',
    'determiner_noun_agreement_with_adj_2': 'determiner-noun agreement',
    'determiner_noun_agreement_with_adj_irregular_1': 'determiner-noun agreement',
    'determiner_noun_agreement_with_adj_irregular_2': 'determiner-noun agreement',

    'ellipsis_n_bar_1': 'ellipsis',
    'ellipsis_n_bar_2': 'ellipsis',

    'wh_questions_object_gap': 'filler-gap',
    'wh_questions_subject_gap': 'filler-gap',
    'wh_questions_subject_gap_long_distance': 'filler-gap',
    'wh_vs_that_no_gap': 'filler-gap',
    'wh_vs_that_no_gap_long_distance': 'filler-gap',
    'wh_vs_that_with_gap': 'filler-gap',
    'wh_vs_that_with_gap_long_distance': 'filler-gap',

    'irregular_past_participle_adjectives': 'irregular forms',
    'irregular_past_participle_verbs': 'irregular forms',

    'distractor_agreement_relational_noun': 'subject-verb agreement',
    'distractor_agreement_relative_clause': 'subject-verb agreement',
    'irregular_plural_subject_verb_agreement_1': 'subject-verb agreement',
    'irregular_plural_subject_verb_agreement_2': 'subject-verb agreement',
    'regular_plural_subject_verb_agreement_1': 'subject-verb agreement',
    'regular_plural_subject_verb_agreement_2': 'subject-verb agreement',

    'existential_there_quantifiers_1': 'quantifiers',
    'existential_there_quantifiers_2': 'quantifiers',
    'superlative_quantifiers_1': 'quantifiers',
    'superlative_quantifiers_2': 'quantifiers',

    'matrix_question_npi_licensor_present': 'npi licensing',
    'npi_present_1': 'npi licensing',
    'npi_present_2': 'npi licensing',
    'only_npi_licensor_present': 'npi licensing',
    'only_npi_scope': 'npi licensing',
    'sentential_negation_npi_licensor_present': 'npi licensing',
    'sentential_negation_npi_scope': 'npi licensing',

    'adjunct_island': 'island effects',
    'complex_NP_island': 'island effects',
    'coordinate_structure_constraint_complex_left_branch': 'island effects',
    'coordinate_structure_constraint_object_extraction': 'island effects',
    'left_branch_island_echo_question': 'island effects',
    'left_branch_island_simple_question': 'island effects',
    'sentential_subject_island': 'island effects',
    'wh_island': 'island effects',

}
phenomena = set([v for v in paradigm2phenomenon.values()])
phenomenon2paradigms = {phenomenon: [pa for pa, ph in paradigm2phenomenon.items() if ph == phenomenon]
                        for phenomenon in phenomena}


if __name__ == '__main__':

    phenomenon2col = defaultdict(list)

    for model_dir in Path('output').glob('*'):

        model_name = model_dir.name
        print('********************************')
        print(model_name)
        print('********************************')

        phenomenon2col['Model'].append(model_name.replace('_', '+'))

        base_dir = Path('output') / model_name

        for phenomenon, paradigms in phenomenon2paradigms.items():

            accuracies = []
            for paradigm in paradigms:
                file = base_dir / f'{paradigm}.txt'
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
                assert num_pairs == 1000
                acc = count / num_pairs
                accuracies.append(acc)

            # collect
            phenomenon_rotated = '\rot{' + phenomenon.capitalize() + '}'
            phenomenon2col[phenomenon_rotated].append(np.mean(accuracies))
            # Since all 67 classes have 1000 pairs, per-class and overall accuracies are the desired (micro)averages

    df = pd.DataFrame(data=phenomenon2col)
    df['Overall'] = df.mean(axis=1)
    print(df.round(2).to_latex(index=False, bold_rows=True, escape=False))

    print()
    for model_name, overall_acc in zip(df['Model'], df['Overall'].round(2)):
        print(f'{model_name:<22} {overall_acc}')
