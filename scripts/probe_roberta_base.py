"""
Probe pre-trained roberta base models
"""
import torch
import shutil


from transformers.models.roberta import RobertaForMaskedLM

from babyberta.utils import load_tokenizer
from babyberta.probing import do_probing
from babyberta import configs
from babyberta.io import save_yaml_file


for model_results_folder_name in ['huggingface_official_base', 'fairseq_official_base']:

    framework, implementation, configuration = model_results_folder_name.split('_')

    # load fairseq roberta base
    if model_results_folder_name.startswith('fairseq'):
        model = torch.hub.load('pytorch/fairseq', 'roberta.base')
        tokenizer = None

    # load huggingface roberta base
    elif model_results_folder_name.startswith('huggingface'):
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        model.cuda(0)
        path_tokenizer_config = configs.Dirs.root / 'data' / 'tokenizers' / 'roberta-base.json'
        tokenizer = load_tokenizer(path_tokenizer_config, max_num_tokens_in_sequence=512)
    else:
        raise AttributeError

    print(f'Num parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    model.eval()

    # remove old results
    path_model_results = configs.Dirs.probing_results / model_results_folder_name
    if path_model_results.exists():
        shutil.rmtree(path_model_results)

    # get step
    if framework == 'fairseq':
        step = 500_000
    elif framework == 'huggingface':
        step = 500_000  # using num steps reported by fairseq (not reported by huggingface)
    else:
        raise AttributeError('Invalid arg to framework')

    # make new save_path
    rep = 0
    save_path = path_model_results / str(rep) / 'saves'
    if not save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)

    # save basic model info
    if not (path_model_results / 'param2val.yaml').exists():
        save_yaml_file(path_out=path_model_results / 'param2val.yaml',
                       param2val={'framework': framework,
                                  'is_official': True,
                                  'is_reference': False,
                                  'is_base': True,
                                  })

    # for each probing task
    for sentences_path in configs.Dirs.probing_sentences.rglob('*.txt'):
        do_probing(save_path,
                   sentences_path,
                   model,
                   step=step,
                   tokenizer=tokenizer,
                   include_punctuation=True,
                   score_with_mask=False,
                   verbose=False
                   )

