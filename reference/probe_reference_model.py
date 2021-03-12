
from transformers import RobertaForMaskedLM

from babybert import configs
from babybert.params import param2default, Params
from babybert.probing import do_probing
from babybert.utils import load_tokenizer


MAX_STEP = 160_000
SCORE_WITH_MASK = False  # true to use pseudo-log likelihoods when scoring sentences in forced choice task

FORCED_CHOICE = False
OPEN_ENDED = True

params = Params.from_param2val(param2default)


if __name__ == '__main__':

    models = []
    steps = []
    names = []

    print(f'Loading reference roberta')
    model_name = f'output/checkpoint-{MAX_STEP}'
    roberta = RobertaForMaskedLM.from_pretrained(model_name)
    roberta.cuda(0)

    print('Loading tokenizer')
    tokenizer = load_tokenizer(params, configs.Dirs.root)

    architecture_name = 'reference_roberta'
    step = MAX_STEP

    # use only 1 rep - and overwrite results each time
    rep = 0
    save_path = configs.Dirs.probing_results / architecture_name / str(rep) / 'saves'

    for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):

        task_type = sentences_path.parent.name

        if not OPEN_ENDED and task_type == 'open_ended':
            continue
        if not FORCED_CHOICE and task_type == 'forced_choice':
            continue

        do_probing(save_path, sentences_path, roberta, tokenizer, step,
                   params.include_punctuation, params.score_with_mask,
                   verbose=True)
