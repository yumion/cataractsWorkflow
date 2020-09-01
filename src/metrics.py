import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt


def confusion_matrix(y_pred, y_true):
    """calculate confusion matrix(tp, fp, fn) with only binary class
    args: shape=(N, H, W)
    return: dict(tp, fp, fn)
    """
    # assert len(y_pred.shape) == 3, f'got {y_pred.shape}'
    # assert len(y_true.shape) == 3, f'got {y_true.shape}'
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp

    return {'tp': tp, 'fp': fp, 'fn': fn}
