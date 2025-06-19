import torch
from .SRNet_L import SRNet_L
from .SRNet_P import generator
from .SRNet_L_fusion import SRNet_L

def model_select(method, pretrained_model_path=None):
    if method == 'SRNet_L':
        model = SRNet_L(in_channels=3,out_channels=32).cuda()
    elif method == 'SRNet_P':
        model = generator().cuda()
    elif method == 'SRNet_L_fusion':
        model = SRNet_L(in_channels=64,out_channels=32).cuda()
    else:
        print(f'Method {method} is not defined!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
