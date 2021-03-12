from pathlib import Path
from typing import List
import numpy as np
import attr
import torch
from torch.nn import CrossEntropyLoss

from tokenizers import Tokenizer
from transformers.models.roberta import RobertaForMaskedLM
from transformers.models.bert import BertForMaskedLM

from babybert.io import save_yaml_file
from babybert.utils import make_sequences
from babybert.dataset import DataSet
from babybert.io import load_sentences_from_file, save_forced_choice_predictions, save_open_ended_predictions


def predict_open_ended(model: RobertaForMaskedLM,
                       dataset: DataSet,
                       ) -> List[str]:
    model.eval()
    res = []

    with torch.no_grad():

        for x, _, mm in dataset:
            # get logits for all words in batch
            output = model(**{k: v.to('cuda') for k, v in attr.asdict(x).items()})
            logits_3d = output['logits'].detach()

            # get predicted words for masked locations
            logits_for_masked_words = logits_3d[mm]  # 2D index into 3D array -> 2D array [num masks, vocab]
            token_ids = [torch.argmax(logits).item() for logits in logits_for_masked_words]
            predicted_words = []
            for i in token_ids:
                w = dataset.tokenizer.id_to_token(i)
                if w is None:
                    raise RuntimeError(f'Did not find token-id={i} in vocab')
                predicted_words.append(w)

            # number of mask symbols should be number of sentences
            if len(predicted_words) != len(logits_3d):
                raise ValueError(f' Num predicted words ({len(predicted_words)}) must be num logits ({len(logits_3d)})')

            res.extend(predicted_words)

    if not res:
        raise RuntimeError('Did not compute predicted words for open_ended task.')

    return res


def predict_forced_choice(model: RobertaForMaskedLM,
                          dataset: DataSet,
                          score_with_mask: bool,
                          ) -> List[float]:
    model.eval()
    cross_entropies = []
    loss_fct = CrossEntropyLoss(reduction='none')

    with torch.no_grad():

        for x, _, _ in dataset:

            if not score_with_mask:
                # get loss
                output = model(**{k: v.to('cuda') for k, v in attr.asdict(x).items()})
                logits_3d = output['logits']
                logits_for_all_words = logits_3d.permute(0, 2, 1)
                labels = x.input_ids.cuda()
                loss = loss_fct(logits_for_all_words,  # need to be [batch size, vocab size, seq length]
                                labels,  # need to be [batch size, seq length]
                                )

                # compute avg cross entropy per sentence
                # to do so, we must exclude loss for padding symbols, using attention_mask
                cross_entropies += [loss_i[np.where(row_mask)[0]].mean().item()
                                    for loss_i, row_mask in zip(loss, x.attention_mask.numpy())]

            else:  # todo test new probing method

                max_token_pos = x.attention_mask.numpy().sum(axis=1).max()
                print(x.attention_mask.numpy())
                print(max_token_pos)
                raise NotImplementedError
                for pos in range(max_token_pos):

                    # insert mask at current position

                    pass


    if not cross_entropies:
        raise RuntimeError('Did not compute cross entropies for forced_choice task.')

    return cross_entropies


def do_probing(save_path: Path,
               sentences_path: Path,
               model: RobertaForMaskedLM,
               tokenizer: Tokenizer,
               step: int,
               include_punctuation: bool,
               score_with_mask: bool,
               verbose: bool = False,
               ) -> None:
    model.eval()

    task_name = sentences_path.stem
    task_type = sentences_path.parent.name

    # load probing sentences into dataset
    print(f'Starting probing with task={task_name}', flush=True)
    sentences = load_sentences_from_file(sentences_path, include_punctuation=include_punctuation)
    sequences = make_sequences(sentences, num_sentences_per_input=1)
    assert len(sequences) == len(sentences)
    probing_dataset = DataSet.for_probing(sequences, tokenizer)

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

    # do inference on forced-choice task
    if task_type == 'forced_choice':
        cross_entropies = predict_forced_choice(model, probing_dataset, score_with_mask)
        save_forced_choice_predictions(sentences, cross_entropies, probing_results_path)

    # do inference on open_ended task
    elif task_type == 'open_ended':
        predicted_words = predict_open_ended(model, probing_dataset)
        save_open_ended_predictions(sentences, predicted_words, probing_results_path,
                                    verbose=True if 'dummy' in task_name else verbose)
    else:
        raise AttributeError('Invalid arg to "task_type".')
