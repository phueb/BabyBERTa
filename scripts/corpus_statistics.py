"""
Count num questions, and num words/sentence for each corpus
"""
import pandas as pd
import numpy as np
from collections import defaultdict

from babyberta.io import load_sentences_from_file
from babyberta.utils import make_sequences, load_tokenizer
from babyberta.dataset import DataSet
from babyberta.params import Params, param2default
from babyberta import configs

# params
params = Params.from_param2val(param2default)

# Byte-level BPE tokenizer
path_tokenizer_config = configs.Dirs.tokenizers / f'{params.tokenizer}.json'
tokenizer = load_tokenizer(path_tokenizer_config, params.max_input_length)

# collect data for data-frame
col2values = defaultdict(list)
for data_path in configs.Dirs.corpora.glob('*.txt'):

    # load sentences
    sentences = load_sentences_from_file(data_path,
                                         include_punctuation=True,
                                         allow_discard=True)

    # make sequences
    sequences = make_sequences(sentences, num_sentences_per_input=1)

    # make dataset (without generating mask patterns to save time)
    dataset = DataSet.for_probing(sequences, tokenizer)

    # compute stats
    num_questions = len([s for s in sentences if s.endswith('?')])
    avg_num_words_per_sentence = np.mean([len(s.split()) for s in sentences])
    avg_num_bpe_tokens_per_sentence = np.mean(dataset.tokenized_sequence_lengths)

    # collect
    corpus_name = data_path.stem
    col2values['Corpus'].append(corpus_name)
    col2values['Sentences'].append(len(sentences))
    col2values['Avg sentence length'].append(avg_num_words_per_sentence)
    col2values['Avg tokenized sentence length'].append(avg_num_bpe_tokens_per_sentence)
    col2values['Questions (proportion)'].append(num_questions / len(sentences))


df = pd.DataFrame(data=col2values)
df.sort_values('Corpus', inplace=True)
df = df.round(2)
print(df)
print(df.to_latex(index=False, bold_rows=True))
