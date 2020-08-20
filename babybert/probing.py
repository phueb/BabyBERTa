from pathlib import Path
from typing import Iterator, Tuple, List
import numpy as np

import torch
from transformers import BertForPreTraining

from babybert import configs
from babybert.io import load_utterances_from_file, save_forced_choice_predictions, save_open_ended_predictions


def predict_forced_choice(model: BertForPreTraining,
                          batches: Iterator,
                          ) -> Tuple[List[List[str]], List[List[str]]]:
    model.eval()

    sentences_in = []
    cross_entropies = []

    for batch in batches:

        with torch.no_grad():
            output = model(**batch)

            raise NotImplementedError

            # sentences_in += ?
            # loss = ?

            # we need 1 loss value per utterance.
            # to do so, we must exclude loss for padding symbols, using attention_mask provided by AllenNLP logic
            loss_cleaned = [row[np.where(row_mask)[0]].mean().item() for row, row_mask in zip(loss, attention_mask)]
            cross_entropies += loss_cleaned
            assert len(sentences_in) == len(cross_entropies)

    return sentences_in, cross_entropies


def predict_open_ended(model: BertForPreTraining,
                       batches: Iterator,
                       ) -> Tuple[List[List[str]], List[List[str]]]:
    model.eval()

    sentences_in = []
    sentences_out = []

    for batch in batches:

        with torch.no_grad():
            output = model(**batch)

            raise NotImplementedError  # TODO

    return sentences_in, sentences_out


def do_probing(task_name: str,
               save_path: Path,
               probing_path: Path,
               model: BertForPreTraining,
               step: int,
               ) -> None:
    for task_type in ['forced_choice', 'open_ended']:

        # prepare data - data is expected to be located on shared drive
        probing_data_path_mlm = probing_path / task_type / f'{task_name}.txt'
        if not probing_data_path_mlm.exists():
            print(f'WARNING: {probing_data_path_mlm} does not exist', flush=True)
            continue
        print(f'Starting probing with task={task_name}', flush=True)
        probing_utterances_mlm = load_utterances_from_file(probing_data_path_mlm)

        batches = None  # TODO

        # prepare out path
        probing_results_path = save_path / task_type / f'probing_{task_name}_results_{step}.txt'
        if not probing_results_path.parent.exists():
            probing_results_path.parent.mkdir(exist_ok=True)

        # do inference on forced-choice task
        if task_type == 'forced_choice':
            sentences_in, cross_entropies = predict_forced_choice(model, batches)
            save_forced_choice_predictions(sentences_in, cross_entropies, probing_results_path)

        # do inference on open_ended task
        elif task_type == 'open_ended':
            sentences_in, sentences_out = predict_open_ended(model, batches)
            save_open_ended_predictions(sentences_in, sentences_out, probing_results_path)

        else:
            raise AttributeError('Invalid arg to "task_type".')