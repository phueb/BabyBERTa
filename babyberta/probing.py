from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from tokenizers import Tokenizer
from transformers.models.roberta import RobertaForMaskedLM


from babyberta.utils import make_sequences
from babyberta.dataset import DataSet
from babyberta.io import load_sentences_from_file, save_forced_choice_predictions, save_open_ended_predictions


RobertaHubInterface = type  # this should be fairseq.RobertaHubInterface but fairseq should not be imported here


def do_probing(save_path: Path,
               sentences_path: Path,
               model: Union[RobertaForMaskedLM, RobertaHubInterface],
               step: int,
               include_punctuation: bool,
               verbose: bool = False,
               tokenizer: Optional[Tokenizer] = None,  # not needed when probing fairseq Roberta
               ) -> None:
    """
    probe a model on a single task.

    a model is a Roberta model, and can be from fairseq or huggingface framework
    """
    model.eval()

    probe_name = sentences_path.stem
    vocab_name = sentences_path.parent.name

    # load probing sentences
    print(f'Starting probing with {probe_name}', flush=True)
    sentences = load_sentences_from_file(sentences_path, include_punctuation=include_punctuation)

    # prepare dataset (if using huggingface model)
    if tokenizer is not None:
        sequences = make_sequences(sentences, num_sentences_per_input=1)
        assert len(sequences) == len(sentences)
        dataset = DataSet.for_probing(sequences, tokenizer)
    else:
        dataset = None

    # prepare out path
    probing_results_path = save_path / vocab_name / f'probing_{probe_name}_results_{step}.txt'
    if not probing_results_path.parent.exists():
        probing_results_path.parent.mkdir(exist_ok=True, parents=True)

    # do inference
    if tokenizer is not None:
        cross_entropies = predict_forced_choice(model, dataset)
    else:
        cross_entropies = predict_forced_choice_fairseq(model, sentences, verbose)

    # save results
    save_forced_choice_predictions(sentences, cross_entropies, probing_results_path)


def predict_forced_choice(model: RobertaForMaskedLM,
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


def predict_forced_choice_fairseq(model: RobertaHubInterface,
                                  sentences: List[str],
                                  verbose: bool,
                                  ):
    from fairseq import utils

    res = []
    loss_fct = CrossEntropyLoss(reduction='none')
    for n, sentence in enumerate(sentences):
        with torch.no_grad():
            tokens = model.encode(sentence).long().to(device=model.device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            with utils.model_eval(model.model):
                features, _ = model.model(tokens,
                                          features_only=False,
                                          return_all_hiddens=False,
                                          )
            logits_3d = features  # [batch size, seq length, vocab size]
            # logits need to be [batch size, vocab size, seq length]
            # tags need to be [batch size, seq length]
            labels = tokens
            ce = loss_fct(logits_3d.permute(0, 2, 1), labels).cpu().numpy().mean()
            res.append(ce)

            if verbose:
                print(
                    f'{n + 1:>6}/{len(sentences):>6} | {ce:.2f} {make_pretty(sentence)}')
    return res
