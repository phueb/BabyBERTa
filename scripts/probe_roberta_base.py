"""
Probe pre-trained roberta base models
"""
import shutil

from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from babyberta.probing import do_probing
from babyberta import configs
from babyberta.io import save_yaml_file


for model_results_folder_name in [
    'huggingface_Roberta-base_10M',
    'huggingface_Roberta-base_30B',
]:

    framework, architecture, data_size = model_results_folder_name.split('_')

    # load NYU roberta-base trained on less data
    if data_size == '10M':
        model = AutoModelForMaskedLM.from_pretrained("nyu-mll/roberta-base-10M-2")
        tokenizer = AutoTokenizer.from_pretrained("nyu-mll/roberta-base-10M-2")

    # load huggingface roberta base
    elif data_size == '30B':
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        raise AttributeError

    print(f'Num parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    model.cuda(0)
    model.eval()

    # remove old results
    path_model_results = configs.Dirs.probing_results / model_results_folder_name
    if path_model_results.exists():
        shutil.rmtree(path_model_results)

    # make new save_path
    rep = 0
    save_path = path_model_results / str(rep) / 'saves'
    if not save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)

    # save basic model info
    if not (path_model_results / 'param2val.yaml').exists():
        save_yaml_file(path_out=path_model_results / 'param2val.yaml',
                       param2val={'framework': framework,
                                  'architecture': architecture,
                                  'data_size': data_size,
                                  })

    # for each probing task
    for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):
        do_probing(save_path,
                   sentences_path,
                   model,
                   step=500_000,
                   tokenizer=tokenizer,
                   include_punctuation=True,
                   verbose=False
                   )

