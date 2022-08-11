import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.data import resolve_data_config, create_transform
from PIL import Image as PILImage
from model.layers import *
from timm.models.factory import create_model

class DnSR50FeatureExtractor(nn.Module):

    def __init__(self, network='resnet50', whiteninig=False, dims=3840):
        super(DnSR50FeatureExtractor, self).__init__()
        self.normalizer = VideoNormalizer()

        self.cnn = models.resnet50(pretrained=True)

        self.rpool = RMAC()
        self.layers = {'layer1': 28, 'layer2': 14, 'layer3': 6, 'layer4': 3}
        if whiteninig or dims != 3840:
            self.pca = PCA(dims)

    def extract_region_vectors(self, x):
        tensors = []
        for nm, module in self.cnn._modules.items():
            if nm not in {'avgpool', 'fc', 'classifier'}:
                x = module(x).contiguous()
                if nm in self.layers:
                    # region_vectors = self.rpool(x)
                    s = self.layers[nm]
                    region_vectors = F.max_pool2d(x, [s, s], int(np.ceil(s / 2)))
                    region_vectors = F.normalize(region_vectors, p=2, dim=1)
                    tensors.append(region_vectors)
        for i in range(len(tensors)):
            tensors[i] = F.normalize(F.adaptive_max_pool2d(tensors[i], tensors[-1].shape[2:]), p=2, dim=1)
        x = torch.cat(tensors, 1)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # (T, h*w, D_sum)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        x = self.normalizer(x)
        x = self.extract_region_vectors(x) # (T, h*w, D_sum)
        if hasattr(self, 'pca'):
            x = self.pca(x) # (T, h*w, 512)
        return x

class TimmFeatureExtractor(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.model = create_model(self.model_name, pretrained=True, num_classes=1000)
        self.model.to(self.device)
        self.model.eval()
        self.config = resolve_data_config({}, model=self.model)
        self.tfms = create_transform(**self.config)

    def forward(self, x: np.ndarray):
        img_tensor_list = []
        for img_np in x:
            img = PILImage.fromarray(img_np.astype('uint8'), 'RGB')
            img_tensor_list.append(self.tfms(img).to(self.device))
        video_tensor = torch.stack(img_tensor_list)
        features = self.model.forward_features(video_tensor)
        if features.dim() == 4:
            global_pool = nn.AdaptiveAvgPool2d(1)
            features = global_pool(features)
        features = features.squeeze()
        return features.detach()