import torch
from efficientnet_pytorch import EfficientNet
from resnest.torch.resnet import ResNet, Bottleneck


"""EfficientNet"""


def efficientnetb0(num_classes, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)


def efficientnetb1(num_classes, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)


def efficientnetb2(num_classes, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)


def efficientnetb3(num_classes, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)


def efficientnetb4(num_classes, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)


def efficientnetb5(num_classes, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)


def efficientnetb6(num_classes, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)


def efficientnetb7(num_classes, **kwargs):
    return EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)


"""ResNeSt"""
_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
]}


def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]


resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
                      name in _model_sha256.keys()}


def resnest50(num_classes, pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64, num_classes=num_classes,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, dilation=2, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model


def resnest101(num_classes, pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64, num_classes=num_classes,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, dilation=2, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model


def resnest200(num_classes, pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64, num_classes=num_classes,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, dilation=2, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model


def resnest269(num_classes, pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64, num_classes=num_classes,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, dilation=2, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model
