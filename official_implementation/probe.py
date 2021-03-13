
from transformers.models.roberta import RobertaForMaskedLM

from babybert import configs
from babybert.params import param2default, Params
from babybert.probing import do_probing
from babybert.utils import load_tokenizer
from babybert.io import save_yaml_file


MAX_STEP = 160_000

FORCED_CHOICE = False
OPEN_ENDED = True

params = Params.from_param2val(param2default)


if __name__ == '__main__':

    for path_model in (configs.Dirs.root / 'official_implementation').glob(f'*/checkpoint-{MAX_STEP}'):
        roberta = RobertaForMaskedLM.from_pretrained(path_model)
        roberta.cuda(0)

        print('Loading tokenizer')
        tokenizer = load_tokenizer(params, configs.Dirs.root)

        model_results_folder_name = 'huggingface_official_reference'
        step = MAX_STEP
        rep = path_model.parent.name
        path_model_results = configs.Dirs.probing_results / model_results_folder_name
        save_path = path_model_results / str(rep) / 'saves'

        # save basic model info
        if not (path_model_results / 'param2val.yaml').exists():
            save_yaml_file(path_out=path_model_results / 'param2val.yaml',
                           param2val={'framework': 'huggingface',
                                      'is_official': 'official',
                                      'is_reference': 'reference'})

        for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):
            do_probing(save_path,
                       sentences_path,
                       roberta,
                       step,
                       params.include_punctuation,
                       params.score_with_mask,
                       tokenizer=tokenizer,
                       verbose=True,
                       )
