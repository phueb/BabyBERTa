import spacy

from babybert import configs

CORPUS_NAME = 'wiki-20191017-hebb-raw'
NUM_MILLION_SEQUENCES = 3

nlp = spacy.load("en_core_web_sm")

in_path = configs.Dirs.root / 'data' / 'corpora' / f'{CORPUS_NAME}.txt'
out_path = configs.Dirs.root / 'data' / 'corpora' / in_path.name.replace('raw', f'{NUM_MILLION_SEQUENCES}M_tokenized')

lines = []
num_periods = 0
with in_path.open('r') as f:
    for n, doc in enumerate(nlp.pipe([l for l in f.readlines()],
                                     disable=['ner', 'tagger', 'parser'])):
        lines.append(' '.join([t.text for t in doc]))

        if num_periods % 100 == 0:
            print(f'{num_periods:,}')

        num_periods += len([t for t in doc if t.is_punct])

        if num_periods > NUM_MILLION_SEQUENCES * 1_000_000:
            break

with out_path.open('w') as f:
    for line in lines:
        f.write(line)