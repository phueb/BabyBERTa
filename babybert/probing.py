from pathlib import Path
from typing import Iterator, Tuple, List
import numpy as np

import torch
from transformers import BertForPreTraining


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