import torch
import models_mae


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16', device='cpu'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=device)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model
