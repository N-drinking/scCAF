import torch
import scCAF
scCAF.load_state_dict(torch.load('./model_pretrained/{}_pretrain.pkl'.format('PBMC-10k'), map_location='cpu'))