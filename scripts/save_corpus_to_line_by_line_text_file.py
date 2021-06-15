"""
This script writes line-by-line text files to disk.

notes:
- aonewsela has 446,672 sentences.
- aochildes has 893,989 sentences
- wikipedia1 has 548,947 sentences
"""


from aochildes.dataset import ChildesDataSet
from aonewsela.dataset import NewselaDataSet

from babyberta.utils import load_wikipedia_sentences
from babyberta import configs

CORPUS_NAME = 'aonewsela'


# load corpus
if CORPUS_NAME == 'aochildes':
    sentences = ChildesDataSet().load_sentences()
elif CORPUS_NAME == 'aonewsela':
    sentences = NewselaDataSet().load_sentences()
elif CORPUS_NAME == 'wikipedia1':
    sentences = load_wikipedia_sentences(configs.Dirs.wikipedia_sentences, percent=20, shift=1)
elif CORPUS_NAME == 'wikipedia2':
    sentences = load_wikipedia_sentences(configs.Dirs.wikipedia_sentences, percent=20, shift=2)
elif CORPUS_NAME == 'wikipedia3':
    sentences = load_wikipedia_sentences(configs.Dirs.wikipedia_sentences, percent=20, shift=3)
else:
    raise AttributeError('Invalid corpus')

print(f'Loaded {len(sentences):,} sentences')

out_path = configs.Dirs.root / 'data' / 'corpora' / f'{CORPUS_NAME}.txt'
if out_path.exists():
    out_path.unlink()  # otherwise, sentences are appended to old file

with out_path.open('w') as f:
    for sentence in sentences:
        f.write(sentence + '\n')
