from pathlib import Path
from typing import Iterator, Tuple, List
import numpy as np

import torch
from transformers import BertForPreTraining, BertTokenizer

from babybert import configs
from babybert.utils import gen_batches
from babybert.io import load_utterances_from_file, save_forced_choice_predictions, save_open_ended_predictions


def predict_open_ended(model: BertForPreTraining,
                       tokenizer: BertTokenizer,
                       sentences: List[List[str]],
                       ) -> List[List[str]]:

    res = []

    for sentences_in_batch in gen_batches(sentences, configs.Eval.batch_size):

        with torch.no_grad():
            batch = tokenizer(sentences_in_batch,
                              padding=True,
                              return_tensors="pt",
                              is_pretokenized=True,
                              return_attention_mask=True)

            # get logits for all words in batch
            output = model(**batch.to('cuda'))
            logits_3d = output[0].detach().cpu().numpy()

            # get predicted words for masked locations
            mask_locations = np.where(batch.data['input_ids'].cpu() == tokenizer.mask_token_id)  # (row ids, col ids)
            logits_for_masked_words = np.squeeze(logits_3d[mask_locations])
            token_ids = [np.argmax(logits).item() for logits in logits_for_masked_words]
            predicted_words = tokenizer.convert_ids_to_tokens(token_ids)
            assert len(predicted_words) == len(logits_3d)

            res += predicted_words

    return res


def predict_forced_choice(model: BertForPreTraining,
                          tokenizer: BertTokenizer,
                          sentences: List[List[str]],
                          ) -> List[float]:
    cross_entropies = []

    for sentences_in_batch in gen_batches(sentences, configs.Eval.batch_size):
        with torch.no_grad():
            batch = tokenizer(sentences_in_batch,
                              padding=True,
                              return_tensors="pt",
                              is_pretokenized=True,
                              return_attention_mask=True)
            output = model(labels=labels, **batch)
            loss = output[0]  # need to provide labels above to get loss as first element in output

            # we need 1 loss value per utterance.
            # to do so, we must exclude loss for padding symbols, using attention_mask
            loss_cleaned = [row[np.where(row_mask)[0]].mean().item() for row, row_mask in zip(loss, attention_mask)]

            raise NotImplementedError  # TODO

    return cross_entropies


def do_probing(task_name: str,
               save_path: Path,
               probing_path: Path,
               tokenizer: BertTokenizer,
               model: BertForPreTraining,
               step: int,
               ) -> None:

    model.eval()

    for task_type in ['forced_choice', 'open_ended']:

        # load probing sentences
        probing_data_path_mlm = probing_path / task_type / f'{task_name}.txt'
        if not probing_data_path_mlm.exists():
            print(f'WARNING: {probing_data_path_mlm} does not exist', flush=True)
            continue
        print(f'Starting probing with task={task_name}', flush=True)
        sentences_in = load_utterances_from_file(probing_data_path_mlm)

        # prepare out path
        probing_results_path = save_path / task_type / f'probing_{task_name}_results_{step}.txt'
        if not probing_results_path.parent.exists():
            probing_results_path.parent.mkdir(exist_ok=True)

        # do inference on forced-choice task
        if task_type == 'forced_choice':

            continue  # TODO remove

            cross_entropies = predict_forced_choice(model, tokenizer, sentences_in)
            save_forced_choice_predictions(sentences_in, cross_entropies, probing_results_path)

        # do inference on open_ended task
        elif task_type == 'open_ended':
            sentences_out = predict_open_ended(model, tokenizer, sentences_in)
            save_open_ended_predictions(sentences_in, sentences_out, probing_results_path)

        else:
            raise AttributeError('Invalid arg to "task_type".')