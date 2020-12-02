from pathlib import Path
from typing import Union, Tuple, List
import numpy as np
import attr

from transformers import RobertaTokenizerFast

import torch
from torch.nn import CrossEntropyLoss
from transformers import BertForPreTraining

from babybert import configs
from babybert.io import save_yaml_file
from babybert.utils import gen_batches, make_sequences
from babybert.io import load_sentences_from_file, save_forced_choice_predictions, save_open_ended_predictions


def predict_open_ended(model: BertForPreTraining,
                       tokenizer: RobertaTokenizerFast,
                       sequences: List[str],
                       ) -> List[str]:
    model.eval()
    res = []

    with torch.no_grad():

        for x, _ in gen_batches(sequences, tokenizer, configs.Eval.batch_size,
                                num_masked=0, consecutive_masking=True):

            # get logits for all words in batch
            output = model(**{k: v.to('cuda') for k, v in attr.asdict(x).items()})
            logits_3d = output[0].detach()

            # get predicted words for masked locations
            mask_locations = x.input_ids == tokenizer.mask_token_id
            logits_for_masked_words = logits_3d[mask_locations]  # 2D index into 3D array -> 2D array [num masks, vocab]
            token_ids = [torch.argmax(logits).item() for logits in logits_for_masked_words]
            predicted_words = tokenizer.convert_ids_to_tokens(token_ids)
            assert len(predicted_words) == len(logits_3d), (len(predicted_words), len(logits_3d))  # number of mask symbols should be number of sentences

            res.extend(predicted_words)

    return res


def predict_forced_choice(model: BertForPreTraining,
                          tokenizer: RobertaTokenizerFast,
                          sequences: List[str],
                          ) -> List[float]:
    model.eval()
    cross_entropies = []
    loss_fct = CrossEntropyLoss(reduction='none')

    with torch.no_grad():

        for x, _ in gen_batches(sequences, tokenizer, configs.Eval.batch_size,
                                num_masked=0, consecutive_masking=True):

            # get loss
            output = model(**{k: v.to('cuda') for k, v in attr.asdict(x).items()})
            logits_3d = output[0]
            logits_for_all_words = logits_3d.permute(0, 2, 1)
            labels = x.input_ids.cuda()
            loss = loss_fct(logits_for_all_words,  # need to be [batch size, vocab size, seq length]
                            labels,  # need to be [batch size, seq length]
                            )

            # compute avg cross entropy per sentence
            # to do so, we must exclude loss for padding symbols, using attention_mask
            cross_entropies += [row[np.where(row_mask)[0]].mean().item()
                                for row, row_mask in zip(loss, x.attention_mask.numpy())]

    return cross_entropies


def do_probing(save_path: Path,
               sentences_in_path: Path,
               tokenizer: RobertaTokenizerFast,
               model: BertForPreTraining,
               step: int,
               include_punctuation: bool,
               ) -> None:

    model.eval()

    task_name = sentences_in_path.stem
    task_type = sentences_in_path.parent.name

    # load probing sentences
    print(f'Starting probing with task={task_name}', flush=True)
    sentences = load_sentences_from_file(sentences_in_path, include_punctuation=include_punctuation)
    sequences = make_sequences(sentences, num_sentences_per_input=1)
    assert len(sequences) == len(sentences)

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
        cross_entropies = predict_forced_choice(model, tokenizer, sequences)
        save_forced_choice_predictions(sentences, cross_entropies, probing_results_path)

    # do inference on open_ended task
    elif task_type == 'open_ended':
        predicted_words = predict_open_ended(model, tokenizer, sequences)
        save_open_ended_predictions(sentences, predicted_words, probing_results_path,
                                    verbose=True if 'dummy' in task_name else False)
    else:
        raise AttributeError('Invalid arg to "task_type".')