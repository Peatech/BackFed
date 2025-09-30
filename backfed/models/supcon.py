import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision.models.vgg import VGG
from torchvision.models.resnet import ResNet  
from .resnet_cifar import ResNet as ResNet_CIFAR
from .resnet_mnist import ResNet as ResNet_MNIST
from .vgg_cifar import VGG as VGG_CIFAR

class SupConModel(nn.Module):
    def __init__(self, model):
        if not isinstance(model, (VGG, ResNet, VGG_CIFAR, ResNet_CIFAR, ResNet_MNIST)):
            raise ValueError("SupConModel only supports VGG and ResNet models")
        
        super(SupConModel, self).__init__()

        self.model = copy.deepcopy(model)
        if isinstance(model, VGG) or isinstance(model, VGG_CIFAR):
            self.model.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
            self.model.classifier = nn.Identity()
        elif isinstance(model, ResNet) or isinstance(model, ResNet_CIFAR) or isinstance(model, ResNet_MNIST):
            self.model.fc = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = F.normalize(x, dim=1)
        return x

    def transfer_params(self, target_model: nn.Module):
        source_params = self.model.state_dict()
        target_params = target_model.state_dict()
        for name, param in source_params.items():
            if name in target_params and target_params[name].shape == param.shape:
                target_params[name].copy_(param.clone())

    def __getattribute__(self, name):
        special_attrs = ["transfer_params", "forward", "model"]
        if name not in special_attrs:
            model = super().__getattribute__('model')
            return getattr(model, name)
        return super().__getattribute__(name)
