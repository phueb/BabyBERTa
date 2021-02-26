"""
The merges.txt file that is the default for fairseq roberta model
needs some small adjustments to be used with huggingface tokenizer:
- replace 'Ä' with 'Ġ'
- remove whitespace between 'Ġ' and previously merged token
- remove second consecutive whitespace

This script makes these adjustments, and can additionally be used to reduce the size of the vocab, by
- excluding special characters, e.g. punctuation, brackets
- excluding numbers
- excluding tokens with uppercase characters

"""
import json
import string

from babybert import configs

EXCLUDE_SPECIAL = True
EXCLUDE_NUMERIC = True
EXCLUDE_UPPER = True

# these are entries added by huggingface and by BabyBERT automatically, independent of merges.txt
start_vocab = {"<long>": 0, "<mask>": 1, "<pad>": 2, "<unk>": 3, "<s>": 4, "</s>": 5, "!": 6, "\"": 7, "#": 8, "$": 9,
               "%": 10, "&": 11, "'": 12, "(": 13, ")": 14, "*": 15, "+": 16, ",": 17, "-": 18, ".": 19, "/": 20,
               "0": 21, "1": 22, "2": 23, "3": 24, "4": 25, "5": 26, "6": 27, "7": 28, "8": 29, "9": 30, ":": 31,
               ";": 32, "<": 33, "=": 34, ">": 35, "?": 36, "@": 37, "A": 38, "B": 39, "C": 40, "D": 41, "E": 42,
               "F": 43, "G": 44, "H": 45, "I": 46, "J": 47, "K": 48, "L": 49, "M": 50, "N": 51, "O": 52, "P": 53,
               "Q": 54, "R": 55, "S": 56, "T": 57, "U": 58, "V": 59, "W": 60, "X": 61, "Y": 62, "Z": 63, "[": 64,
               "\\": 65, "]": 66, "^": 67, "_": 68, "`": 69, "a": 70, "b": 71, "c": 72, "d": 73, "e": 74, "f": 75,
               "g": 76, "h": 77, "i": 78, "j": 79, "k": 80, "l": 81, "m": 82, "n": 83, "o": 84, "p": 85, "q": 86,
               "r": 87, "s": 88, "t": 89, "u": 90, "v": 91, "w": 92, "x": 93, "y": 94, "z": 95, "{": 96, "|": 97,
               "}": 98, "~": 99, "¡": 100, "¢": 101, "£": 102, "¤": 103, "¥": 104, "¦": 105, "§": 106, "¨": 107,
               "©": 108, "ª": 109, "«": 110, "¬": 111, "®": 112, "¯": 113, "°": 114, "±": 115, "²": 116, "³": 117,
               "´": 118, "µ": 119, "¶": 120, "·": 121, "¸": 122, "¹": 123, "º": 124, "»": 125, "¼": 126, "½": 127,
               "¾": 128, "¿": 129, "À": 130, "Á": 131, "Â": 132, "Ã": 133, "Ä": 134, "Å": 135, "Æ": 136, "Ç": 137,
               "È": 138, "É": 139, "Ê": 140, "Ë": 141, "Ì": 142, "Í": 143, "Î": 144, "Ï": 145, "Ð": 146, "Ñ": 147,
               "Ò": 148, "Ó": 149, "Ô": 150, "Õ": 151, "Ö": 152, "×": 153, "Ø": 154, "Ù": 155, "Ú": 156, "Û": 157,
               "Ü": 158, "Ý": 159, "Þ": 160, "ß": 161, "à": 162, "á": 163, "â": 164, "ã": 165, "ä": 166, "å": 167,
               "æ": 168, "ç": 169, "è": 170, "é": 171, "ê": 172, "ë": 173, "ì": 174, "í": 175, "î": 176, "ï": 177,
               "ð": 178, "ñ": 179, "ò": 180, "ó": 181, "ô": 182, "õ": 183, "ö": 184, "÷": 185, "ø": 186, "ù": 187,
               "ú": 188, "û": 189, "ü": 190, "ý": 191, "þ": 192, "ÿ": 193, "Ā": 194, "ā": 195, "Ă": 196, "ă": 197,
               "Ą": 198, "ą": 199, "Ć": 200, "ć": 201, "Ĉ": 202, "ĉ": 203, "Ċ": 204, "ċ": 205, "Č": 206, "č": 207,
               "Ď": 208, "ď": 209, "Đ": 210, "đ": 211, "Ē": 212, "ē": 213, "Ĕ": 214, "ĕ": 215, "Ė": 216, "ė": 217,
               "Ę": 218, "ę": 219, "Ě": 220, "ě": 221, "Ĝ": 222, "ĝ": 223, "Ğ": 224, "ğ": 225, "Ġ": 226, "ġ": 227,
               "Ģ": 228, "ģ": 229, "Ĥ": 230, "ĥ": 231, "Ħ": 232, "ħ": 233, "Ĩ": 234, "ĩ": 235, "Ī": 236, "ī": 237,
               "Ĭ": 238, "ĭ": 239, "Į": 240, "į": 241, "İ": 242, "ı": 243, "Ĳ": 244, "ĳ": 245, "Ĵ": 246, "ĵ": 247,
               "Ķ": 248, "ķ": 249, "ĸ": 250, "Ĺ": 251, "ĺ": 252, "Ļ": 253, "ļ": 254, "Ľ": 255, "ľ": 256, "Ŀ": 257,
               "ŀ": 258, "Ł": 259, "ł": 260, "Ń": 261}


def prepare_for_filtering(new_line: str):
    """
    convert line from merges.txt to string of characters eligible for filtering.

    note: removes 'Ġ' because it is non-ASCII and one filter option is to exclude non-ASCII characters.
    """
    res = ''.join([i for i in new_line if i not in {'Ġ', ' '}])
    return res


if __name__ == '__main__':
    # load merges
    path_in = configs.Dirs.tokenizers / 'gpt2_bpe' / 'merges.txt'
    old_lines = path_in.read_text()

    # load vocab
    vocab_path = configs.Dirs.tokenizers / 'gpt2_bpe' / 'vocab.json'
    vocab = json.load(vocab_path.open('r'))

    num_dropped = 0
    num_bad_merges = 0
    new_lines = []
    new_vocab = start_vocab
    for ol in old_lines.split('\n'):
        if ol.startswith('#'):
            continue

        nl = ol.replace('Ä', 'Ġ').replace('  ', ' ')

        # remove whitespace between previously merged tokens
        if nl.count(' ') == 2:
            nl = nl.replace(' ', '', 1)  # remove first occurrence only

        print()
        print('Line in merges.txt:')
        print(nl)

        chars = prepare_for_filtering(nl)

        if [i for i in nl.split() if i not in vocab]:  # individual token components must be in vocab
            num_bad_merges += 1
            continue

        if [i for i in chars if i.isdigit()]:
            num_dropped += 1
            continue

        if [i for i in chars if i in string.punctuation] and EXCLUDE_SPECIAL:
            num_dropped += 1
            continue

        if [i for i in chars if i.isupper()] and EXCLUDE_UPPER:
            num_dropped += 1
            continue

        # collect vocab
        print('Adding to vocab:')
        merged = ''.join(nl.replace(' ', ''))
        new_vocab.setdefault(merged, len(new_vocab))
        print(merged)

        # collect merges
        new_lines.append(nl)

    print(f'Dropped  {num_dropped:,} tokens from original vocab')
    print(f'Found    {num_bad_merges:,} bad merges')
    print(f'Included {len(new_vocab):,} in new vocab')

    path_out = configs.Dirs.tokenizers / 'gpt2_bpe' / 'merges_converted.txt'
    with path_out.open('w', encoding='utf-8') as f:
        for nl in new_lines:
            # print(nl)
            f.write(nl + '\n')

    path_out = configs.Dirs.tokenizers / 'gpt2_bpe' / 'vocab_converted.json'
    json.dump(new_vocab, path_out.open('w'))
