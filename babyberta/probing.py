from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from tokenizers import Tokenizer
from transformers.models.roberta import RobertaForMaskedLM


from babyberta.utils import make_sequences
from babyberta.dataset import DataSet
from babyberta.io import load_sentences_from_file, save_forced_choice_predictions


def do_probing(save_path: Path,
               paradigm_path: Path,
               model: RobertaForMaskedLM,
               step: int,
               include_punctuation: bool,
               tokenizer: Tokenizer,
               ) -> None:
    """
    probe a model on a single paradigm.
    """
    model.eval()

    probe_name = paradigm_path.stem
    vocab_name = paradigm_path.parent.name

    # load probing sentences
    print(f'Starting probing with {probe_name}', flush=True)
    sentences = load_sentences_from_file(paradigm_path, include_punctuation=include_punctuation)

    # prepare dataset
    sequences = make_sequences(sentences, num_sentences_per_input=1)
    assert len(sequences) == len(sentences)
    dataset = DataSet.for_probing(sequences, tokenizer)

    # prepare out path
    probing_results_path = save_path / vocab_name / f'probing_{probe_name}_results_{step}.txt'
    if not probing_results_path.parent.exists():
        probing_results_path.parent.mkdir(exist_ok=True, parents=True)

    # do inference
    cross_entropies = calc_cross_entropies(model, dataset)

    # save results  (save non-lower-cased sentences)
    save_forced_choice_predictions(sentences, cross_entropies, probing_results_path)


def calc_cross_entropies(model: RobertaForMaskedLM,
                         dataset: DataSet,
                         ) -> List[float]:
    model.eval()
    cross_entropies = []
    loss_fct = CrossEntropyLoss(reduction='none')

    with torch.no_grad():

        for x, _, _ in dataset:

            # get loss
            output = model(**{k: v.to('cuda') for k, v in x.items()})
            logits_3d = output['logits']
            logits_for_all_words = logits_3d.permute(0, 2, 1)
            labels = x['input_ids'].cuda()
            loss = loss_fct(logits_for_all_words,  # need to be [batch size, vocab size, seq length]
                            labels,  # need to be [batch size, seq length]
                            )

            # compute avg cross entropy per sentence
            # to do so, we must exclude loss for padding symbols, using attention_mask
            cross_entropies += [loss_i[np.where(row_mask)[0]].mean().item()
                                for loss_i, row_mask in zip(loss, x['attention_mask'].numpy())]

    if not cross_entropies:
        raise RuntimeError(f'Did not compute cross entropies.')

    return cross_entropies


def make_pretty(sentence: str):
    return " ".join([f"{w:<18}" for w in sentence.split()])
