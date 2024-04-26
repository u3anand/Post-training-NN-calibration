"""
Model Library: Resnet 18 and Resnet34
"""
import torch.nn as nn
import torchvision.models as models

def get_resnet18_model(num_classes=100):
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

def get_resnet34_model(num_classes=100):
    model = models.resnet34(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model