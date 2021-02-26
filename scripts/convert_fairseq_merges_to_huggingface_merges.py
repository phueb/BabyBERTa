"""
The merges.txt file that is the default for fairseq roberta model
needs some small adjustments to be used tih huggingface tokenizer.

This script makes these adjustments
"""


from babybert import configs


if __name__ == '__main__':
    path_in = configs.Dirs.tokenizers / 'gpt2_bpe' / 'merges.txt'
    old_lines = path_in.read_text()

    path_in.rename(path_in.parent / 'merges_original.txt')

    new_lines = []
    for ol in old_lines.split('\n'):
        nl = ol.replace('Ä', 'Ġ').replace('  ', ' ')

        # remove whitespace between previously merged tokens
        if nl.count(' ') == 2:
            nl = nl.replace(' ', '', 1)

        new_lines.append(nl)

    path_out = configs.Dirs.tokenizers / 'gpt2_bpe' / 'merges.txt'
    with path_out.open('w') as f:
        for nl in new_lines:
            f.write(nl + '\n')

