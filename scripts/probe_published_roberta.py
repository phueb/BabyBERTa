import torch
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta import RobertaModel

from babybert.probing import do_probing



# load fairseq roberta
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
print(f'Loaded roberta.base from torch.hub')




# loa huggingface robertaa # TODO