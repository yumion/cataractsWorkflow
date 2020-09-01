import torch
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest50, resnest101, resnest200, resnest269


def EfficientNetB0(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)


def EfficientNetB1(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)


def EfficientNetB2(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)


def EfficientNetB3(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)


def EfficientNetB4(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)


def EfficientNetB5(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)


def EfficientNetB6(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)


def EfficientNetB7(num_classes):
    return EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
