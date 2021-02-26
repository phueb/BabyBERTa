"""
The merges.txt file that is the default for fairseq roberta model
needs some small adjustments to be used tih huggingface tokenizer.

This script makes these adjustments
"""
import json

from babybert import configs

ALLOW_DROPPING = True


if __name__ == '__main__':
    # load merges
    path_in = configs.Dirs.tokenizers / 'gpt2_bpe' / 'merges.txt'
    old_lines = path_in.read_text()

    # load vocab
    vocab_path = configs.Dirs.tokenizers / 'gpt2_bpe' / 'vocab.json'
    vocab = json.load(vocab_path.open('r'))

    num_dropped= 0
    new_lines = []
    for ol in old_lines.split('\n'):
        if ol.startswith('#'):
            continue

        nl = ol.replace('Ä', 'Ġ').replace('  ', ' ')

        # remove whitespace between previously merged tokens
        if nl.count(' ') == 2:
            nl = nl.replace(' ', '', 1)

        # check vocab
        token = nl.replace(' ', '')
        if token not in vocab:
            e = f'"{token}" not in vocab.json'
            if ALLOW_DROPPING:
                print(e)
                num_dropped += 1
                continue
            else:
                raise KeyError(e)


        new_lines.append(nl)

    print(f'Found {num_dropped} merges not in vocab.json')

    path_out = configs.Dirs.tokenizers / 'gpt2_bpe' / 'merges_converted.txt'
    with path_out.open('w') as f:
        for nl in new_lines:
            f.write(nl + '\n')

