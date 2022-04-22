import numpy as np
import torch
from torch import nn

from Stereo_matching.NeuralNetwork.ACVNet_main.models import __models__
from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_main import test_sample


def ACVNet_test(sample, max_disp, verbose):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    model = __models__['acvnet'](192, False, False)
    model = nn.DataParallel(model)
    model.cuda()
    print(f"loading model {'pretrained_model_sceneflow.ckpt'}")
    state_dict = torch.load('Stereo_matching/NeuralNetwork/ACVNet_main/pretrained_model/pretrained_model_sceneflow.ckpt')
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
    disp_available = False
    image_outputs = test_sample(sample, model, max_disp, compute_metrics=disp_available, verbose=verbose)
    m, M = np.array(image_outputs['disp_est']).min(), np.array(image_outputs['disp_est']).max()
    if verbose:
        print(f"Max disparity = {M}\n"
              f"Min disparity = {m}")
    image_outputs['disp_est'] = (image_outputs['disp_est'] - m) / (M - m)
    return image_outputs['imgL'], image_outputs['imgR'], image_outputs['disp_est'], m, M