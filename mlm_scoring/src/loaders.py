""""
Adapted from https://github.com/awslabs/mlm-scoring
"""
from collections import defaultdict, OrderedDict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple
import gluonnlp as nlp
import numpy as np


class Hypotheses():
    """A wrapper around hypotheses for a single utterance
    """

    def __init__(self, sents: List[str], scores: List[float], vocab: Optional[nlp.Vocab] = None, tokenizer: Any = None):
        self.sents = sents
        self.scores = scores
        self.tokenizer = tokenizer
        self._vocab = vocab

    def _generate_ln(self, alpha=0.6, tokenizer=None, ln_type='gnmt'):
        self.sent_lens = np.zeros(shape=(len(self.sents),))
        for idx, sent in enumerate(self.sents):
            if ln_type == 'gnmt':
                self.sent_lens[idx] = (5 + len(tokenizer(sent)))**alpha / (5 + 1)**alpha
            elif ln_type == 'length':
                self.sent_lens[idx] = len(tokenizer(sent))
            else:
                raise ValueError("Invalid length normalization type '{}'".format(ln_type))
        return self.sent_lens


class Corpus(OrderedDict):
    """A ground truth corpus (dictionary of ref sentences)
    """

    @classmethod
    def from_file(cls, fp: TextIO, **kwargs) -> Any:
        # A text file (probably LM training data per line)
        return Corpus.from_text(fp, **kwargs)

    @classmethod
    def from_text(cls, fp: Iterable[str], lower_case=False):
        corpus = cls()
        # For text files, utterance ID is just the zero-indexed line number
        idx = 0
        for line in fp:
            if lower_case:
                corpus[idx] = line.strip().lower()
            else:
                corpus[idx] = line.strip()

            idx += 1
        return corpus

    def _get_num_words_in_utt(self, utt: str) -> int:
        return len(utt.split(' '))

    def get_num_words(self) -> Tuple[List[int], int]:

        max_words_per_utt = 0
        num_words_list = []

        for utt_id, utt in self.items():
            num_words = self._get_num_words_in_utt(utt)
            num_words_list.append(num_words)
            if num_words > max_words_per_utt:
                max_words_per_utt = num_words

        return num_words_list, max_words_per_utt


class Predictions(OrderedDict):
    """A dictionary of hypotheses
    """

    # When "deserializing" a corpus into predictions, what token separates the utt id from the number?
    SEPARATOR = '--'


    @classmethod
    def from_file(cls, fp: TextIO, **kwargs):
        suffix = Path(fp.name).suffix
        if suffix == '.json':
            obj_dict = json.load(fp)
            return cls.from_dict(obj_dict, **kwargs)
        else:
            raise ValueError("Hypothesis file of type '{}' is not supported".format(suffix))

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Dict[str, Any]], max_utts: Optional[int] = None, vocab: Optional[nlp.Vocab] = None, tokenizer = None):
        """Loads hypotheses from the format of Shin et al. (JSON)

        Args:
            fp (str): JSON file name
            max_utts (None, optional): Number of utterances to process
            vocab (None, optional): Vocabulary

        Returns:
            TYPE: Description
        """

        # Just a dictionary for now
        # but equipped with this factory method
        preds = cls()

        item_list = sorted(obj_dict.items())
        if max_utts is not None:
            item_list = item_list[:max_utts]
        for utt_id, hyps_dict in item_list:

            num_hyps = 0
            for key in hyps_dict.keys():
                if key.startswith("hyp_"):
                    num_hyps += 1

            sents = [None]*num_hyps
            scores = [None]*num_hyps
            for hyp_id, hyp_data in hyps_dict.items():
                if not hyp_id.startswith('hyp_'):
                    continue
                # 'hyp_100' --> 99
                idx = int(hyp_id.split('_')[1]) - 1
                sents[idx] = hyp_data['text'].strip()
                scores[idx] = hyp_data['score']

            hyps = Hypotheses(sents, scores, vocab, tokenizer)
            preds[utt_id] = hyps

        return preds

    def to_corpus(self) -> Corpus:

        corpus = Corpus()
        for utt_id, hyps in self.items():
            for idx, sent in enumerate(hyps.sents):
                corpus["{}{}{}".format(utt_id, self.SEPARATOR, idx+1)] = sent

        return corpus

    def to_json(self, fp: TextIO):

        json_dict = {}

        for utt_id, hyps in self.items():

            json_dict[utt_id] = {}

            for idx, (sent, score) in enumerate(zip(hyps.sents, hyps.scores)):
                json_dict[utt_id]["hyp_{}".format(idx+1)] = {
                    "score": float(score),
                    "text": sent
                }

        json.dump(json_dict, fp, indent=2, separators=(',', ': '), sort_keys=True)


class ScoredCorpus(OrderedDict):

    @classmethod
    def from_corpus_and_scores(cls, corpus: Corpus, scores: List[float]) -> OrderedDict:
        scored_corpus = ScoredCorpus()
        for (idx, text), score in zip(corpus.items(), scores):
            scored_corpus[idx] = {'score': score, 'text': text}
        return scored_corpus

    def to_file(self,
                file_path: Path,
                scores_only: bool = False):

        fp = file_path.open('w')

        for idx, data in self.items():
            # Either a float or a list of floats/null
            line = "{}\n".format(json.dumps(data['score']))
            if not scores_only:
                line = "{} ".format(data['text']) + line
            fp.write(line)

    def to_predictions(self) -> OrderedDict: 

        hyp_dict = defaultdict(dict)

        for key, val in self.items():
            data_key = key.split(Predictions.SEPARATOR)
            if not (len(data_key) == 2 and data_key[1].isdigit()):
                raise ValueError("This ScoredCorpus cannot be deserialized into Predictions")
            utt_id = data_key[0]
            hyp_num = int(data_key[1])
            hyp_dict[utt_id]['hyp_{}'.format(hyp_num)] = val

        return Predictions.from_dict(hyp_dict)
