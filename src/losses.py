import torch
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.utils import base
from utils import take_channels


class L1Loss(torch.nn.L1Loss, base.Loss):
    pass


class MSELoss(torch.nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(torch.nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(torch.nn.NLLLoss, base.Loss):
    pass


class BCELoss(torch.nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss, base.Loss):
    pass


class CategoricalFocalLoss(base.Loss):
    r"""Creates a criterion that measures the Categorical Focal Loss between the
    ground truth (gt) and the prediction (pr).
    .. math:: L(gt, pr) = - gt \cdot \alpha \cdot (1 - pr)^\gamma \cdot \log(pr)
    Args:
        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.
        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    Returns:
        A callable ``categorical_focal_loss`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.
    """

    def __init__(self, alpha=0.25, gamma=2., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, pr, gt):
        pr = self.activation(pr)
        return categorical_focal_loss(
            pr, gt,
            alpha=self.alpha,
            gamma=self.gamma,
            ignore_channels=self.ignore_channels,
        )


class OnehotCrossEntropyLoss(base.Loss):

    def __init__(self, class_weights=None, reduction='mean', activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.reduction = reduction
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, pr, gt):
        pr = self.activation(pr)
        return categorical_crossentropy_loss(
            pr, gt,
            class_weights=self.class_weights,
            reduction=self.reduction,
            ignore_channels=self.ignore_channels,
        )


def categorical_focal_loss(pr, gt, gamma=2.0, alpha=0.25, ignore_channels=None, eps=1e-7, **kwargs):
    """Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    Args:
        gt: ground truth 4D tensor (B, C, H, W)
        pr: prediction 4D tensor (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        ignore_channels: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    """

    gt, pr = take_channels(gt, pr, ignore_channels=ignore_channels, **kwargs)
    # clip to prevent NaN's and Inf's
    pr = torch.clamp(pr, eps, 1.0 - eps)
    # Calculate focal loss
    loss = - gt * (alpha * torch.pow((1 - pr), gamma) * torch.log(pr + eps))

    return torch.mean(loss)


def categorical_crossentropy_loss(pr, gt, class_weights=None, reduction='mean', ignore_channels=None, eps=1e-7, **kwargs):
    """CrossEntropy loss with onehot-encoding labels
    """
    gt, pr = take_channels(gt, pr, ignore_channels=ignore_channels, **kwargs)
    # cast onehot to class id
    gt = torch.argmax(gt, axis=1)
    # build instance of torch module
    loss = torch.nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    return loss(pr, gt)
