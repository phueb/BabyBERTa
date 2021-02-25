"""
This script tokenizes a custom-made Wiki corpus using spacy,
 so that it matches the tokenization used for the other two corpora (e.g. CHILDES, Newsela).
It also outputs a corpus that roughly matches the other two in the number of sequences.
"""
import spacy
from pathlib import Path

from babybert import configs

CORPUS_PATH = Path('/media/ludwig_data/CreateWikiCorpus/runs/param_22/hebb_2019-10-17-23-02-53_num0/more_words_small_bodies.txt')
OUT_NAME = 'wiki-20191017-hebb'
NUM_MILLION_SEQUENCES = 3

nlp = spacy.load("en_core_web_sm")


lines = []
num_periods = 0
with CORPUS_PATH.open('r') as f:
    for n, doc in enumerate(nlp.pipe([l for l in f.readlines()],
                                     disable=['ner', 'tagger', 'parser'])):
        lines.append(' '.join([t.text for t in doc]))

        if num_periods % 100 == 0:
            print(f'{num_periods:,}')

        num_periods += len([t for t in doc if t.is_punct])

        if num_periods > NUM_MILLION_SEQUENCES * 1_000_000:
            break

out_path = configs.Dirs.root / 'data' / 'corpora' / f'{OUT_NAME}-{NUM_MILLION_SEQUENCES}M_tokenized.txt'
with out_path.open('w') as f:
    for line in lines:
        f.write(line)
