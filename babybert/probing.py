from pathlib import Path
from typing import Union, Tuple, List
import numpy as np

from transformers import RobertaTokenizerFast
from transformers.modeling_roberta import create_position_ids_from_input_ids

import torch
from torch.nn import CrossEntropyLoss
from transformers import BertForPreTraining, BertTokenizerFast

from babybert import configs
from babybert.io import save_yaml_file
from babybert.utils import gen_batches_without_labels
from babybert.io import load_utterances_from_file, save_forced_choice_predictions, save_open_ended_predictions


def predict_open_ended(model: BertForPreTraining,
                       tokenizer: Union[RobertaTokenizerFast, BertTokenizerFast],
                       sentences: List[List[str]],
                       ) -> List[List[str]]:

    res = []

    for sentences_in_batch in gen_batches_without_labels(sentences, configs.Eval.batch_size):

        with torch.no_grad():
            if isinstance(tokenizer, RobertaTokenizerFast):
                tokenizer_input = [' '.join(s) for s in sentences_in_batch]
            else:
                tokenizer_input = sentences_in_batch
            batch = tokenizer.batch_encode_plus(tokenizer_input,
                                                padding=True,
                                                return_tensors='pt')
            position_ids = create_position_ids_from_input_ids(batch.data['input_ids'], tokenizer.pad_token_id)

            # get logits for all words in batch
            output = model(**batch.to('cuda'), position_ids=position_ids.to('cuda'))
            logits_3d = output[0].detach()

            # get predicted words for masked locations
            mask_locations = batch.data['input_ids'] == tokenizer.mask_token_id
            logits_for_masked_words = logits_3d[mask_locations]  # 2D index into 3D array -> 2D array [batch, vocab]
            token_ids = [torch.argmax(logits).item() for logits in logits_for_masked_words]
            predicted_words = tokenizer.convert_ids_to_tokens(token_ids)
            assert len(predicted_words) == len(logits_3d), (len(predicted_words), len(logits_3d))  # number of mask symbols should be number of sentences

            res += predicted_words

    return res


def predict_forced_choice(model: BertForPreTraining,
                          tokenizer: Union[RobertaTokenizerFast, BertTokenizerFast],
                          sentences: List[List[str]],
                          ) -> List[float]:
    cross_entropies = []
    loss_fct = CrossEntropyLoss(reduction='none')

    for sentences_in_batch in gen_batches_without_labels(sentences, configs.Eval.batch_size):
        with torch.no_grad():
            if isinstance(tokenizer, RobertaTokenizerFast):
                tokenizer_input = [' '.join(s) for s in sentences_in_batch]
            else:
                tokenizer_input = sentences_in_batch
            batch = tokenizer.batch_encode_plus(tokenizer_input,
                                                padding=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
            position_ids = create_position_ids_from_input_ids(batch.data['input_ids'], tokenizer.pad_token_id)

            # logits
            output = model(**batch.to('cuda'), position_ids=position_ids.to('cuda'))
            logits_3d = output[0]

            # compute avg cross entropy per sentence
            labels = batch.data['input_ids']
            # logits need to be [batch size, vocab size, seq length]
            # tags need to be [batch size, vocab size]
            loss = loss_fct(logits_3d.permute(0, 2, 1), labels)

            # we need 1 loss value per utterance.
            # to do so, we must exclude loss for padding symbols, using attention_mask
            cross_entropies += [row[np.where(row_mask)[0]].mean().item()
                                for row, row_mask in zip(loss, batch.data['attention_mask'].cpu().numpy())]

    return cross_entropies


def do_probing(save_path: Path,
               sentences_in_path: Path,
               tokenizer: Union[RobertaTokenizerFast, BertTokenizerFast],
               model: BertForPreTraining,
               step: int,
               include_punctuation: bool,
               ) -> None:

    model.eval()

    task_name = sentences_in_path.stem
    task_type = sentences_in_path.parent.name

    # load probing sentences
    print(f'Starting probing with task={task_name}', flush=True)
    sentences_in = load_utterances_from_file(sentences_in_path, include_punctuation=include_punctuation)

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
        cross_entropies = predict_forced_choice(model, tokenizer, sentences_in)
        save_forced_choice_predictions(sentences_in, cross_entropies, probing_results_path)

    # do inference on open_ended task
    elif task_type == 'open_ended':
        sentences_out = predict_open_ended(model, tokenizer, sentences_in)
        save_open_ended_predictions(sentences_in, sentences_out, probing_results_path,
                                    verbose=True if 'dummy' in task_name else False)
    else:
        raise AttributeError('Invalid arg to "task_type".')