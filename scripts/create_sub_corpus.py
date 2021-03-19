"""
This script outputs a corpus with a requested number of sequences.
"""
from pathlib import Path

from babyberta import configs

CORPUS_PATH = ''
NUM_MILLION_SEQUENCES = 3


lines = []
num_punctuation = 0
with Path(CORPUS_PATH).open('r') as f:
    docs = [l for l in f.readlines()]
    for n, doc in enumerate(docs):
        lines.append(' '.join([t for t in doc]))

        if num_punctuation % 100 == 0:
            print(f'{num_punctuation:,}')

        num_punctuation += len([t for t in doc if t in {'.', '!', '?'}])

        if num_punctuation > NUM_MILLION_SEQUENCES * 1_000_000:
            break

out_path = configs.Dirs.root / 'data' / 'corpora' / f'{Path(CORPUS_PATH).stem}-{NUM_MILLION_SEQUENCES}M.txt'
with out_path.open('w') as f:
    for line in lines:
        f.write(line)
