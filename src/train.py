import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Union, Optional
import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import albumentations as albu
from segmentation_models_pytorch import utils

from dataset import ClassificationVideoDataset
import models
import losses
from utils import get_preprocessing, plot_logs, normalization
from augmentations import get_augmentation_wrapper


def main(config):
    # set dataset path
    x_train_dir = [os.path.join(config.DATA_DIR, train_dir) for train_dir in config.TRAINING_DIRS]
    x_valid_dir = [os.path.join(config.DATA_DIR, train_dir) for train_dir in config.VALIDATION_DIRS]

    # create segmentation model with pretrained encoder
    model = getattr(models, config.MODEL)(
        classes=config.N_CLASSES,
    )
    # model = torch.nn.DataParallel(model)

    # create loss function
    loss = getattr(losses, config.LOSS)(
        **config.loss_params
    )

    # set metrics
    metrics = [
        utils.metrics.Accuracy(),
        utils.metrics.Precision(),
        utils.metrics.Recall(),
        utils.metrics.Fscore(),
        utils.metrics.IoU(),
    ]

    # set optimizer
    optimizer = getattr(torch.optim, config.OPTIMIZER)(
        params=model.parameters(),
        lr=config.LR,
        **config.optim_params,
    )
    # learning rate scheduler
    scheduler = getattr(torch.optim.lr_scheduler, config.SCHEDULER)(
        optimizer,
        **config.scheduler_params
    )

    # create Dataset and DataLoader
    preprocessing_fn = normalization

    # set augmentation
    training_augmentation = get_augmentation_wrapper(config.TRAINING_AUGORATION)
    validation_augmentation = get_augmentation_wrapper(config.VALIDATION_AUGORATION)

    train_dataset = ClassificationVideoDataset(
        x_train_dir,
        augmentation=training_augmentation(height=config.HEIGHT, width=config.WIDTH),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
        skip_frame=config.SKIP_FRAME,
    )
    print('number of train data:', len(train_dataset))

    valid_dataset = ClassificationVideoDataset(
        x_valid_dir,
        augmentation=validation_augmentation(height=config.HEIGHT, width=config.WIDTH),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
        skip_frame=config.SKIP_FRAME,
    )
    print('number of validation data:', len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # create epoch runners
    # it is a simple loop of iterating over dataloader's samples
    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )

    # train model
    loggers = {
        'epoch': [],
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': [],
        'precision': [],
        'val_precision': [],
        'recall': [],
        'val_recall': [],
        'fscore': [],
        'val_fscore': [],
        'jaccard': [],
        'val_jaccard': [],
    }

    # callbacks用
    max_score = 0

    for i in range(0, config.EPOCHS):

        print('\nEpoch: {}'.format(i + 1))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        ## logger ##
        # TODO: 冗長すぎる
        loggers['epoch'].append(i + 1)
        loggers['loss'].append(train_logs['loss'])
        loggers['val_loss'].append(valid_logs['loss'])
        loggers['accuracy'].append(train_logs['accuracy'])
        loggers['val_accuracy'].append(valid_logs['accuracy'])
        loggers['precision'].append(train_logs['precision'])
        loggers['val_precision'].append(valid_logs['precision'])
        loggers['recall'].append(train_logs['recall'])
        loggers['val_recall'].append(valid_logs['recall'])
        loggers['fscore'].append(train_logs['fscore'])
        loggers['val_fscore'].append(valid_logs['fscore'])
        loggers['jaccard'].append(train_logs['iou_score'])
        loggers['val_jaccard'].append(valid_logs['iou_score'])

        # save logs to csv
        df = pd.DataFrame(loggers)
        df.to_csv(os.path.join(config.RESULT_DIR, 'logs.csv'), index=False)

        # plot logs
        plot_logs(loggers, 'loss', config.RESULT_DIR)
        plot_logs(loggers, 'accuracy', config.RESULT_DIR)
        plot_logs(loggers, 'precision', config.RESULT_DIR)
        plot_logs(loggers, 'recall', config.RESULT_DIR)
        plot_logs(loggers, 'fscore', config.RESULT_DIR)
        plot_logs(loggers, 'jaccard', config.RESULT_DIR)

        ## callbacks (save model, change lr, etc.) ##
        # model checkpoint
        if max_score < valid_logs['fscore']:
            max_score = valid_logs['fscore']
            torch.save(model, os.path.join(config.RESULT_DIR, 'best_model.pth'))
            print('Model saved.')

        # learning rate schedule
        if config.SCHEDULER == 'ReduceLROnPlateau':
            scheduler.step(valid_logs['loss'])
        else:
            scheduler.step()

    # save last epoch model
    torch.save(model, os.path.join(config.RESULT_DIR, 'last_model.pth'))


@dataclass
class Config:
    DEVICE: str = 'cuda'
    SEED: int = 1

    RESULT_DIR: str = '/data1/github/MICCAI2020/cataractsWorkflow/result/cnn_only'

    DATA_DIR: str = '/data1/github/MICCAI2020/cataractsWorkflow/data'
    TRAINING_DIRS: list = field(default_factory=list)
    VALIDATION_DIRS: list = field(default_factory=list)

    CLASSES: Union[list, dict] = field(default_factory=dict)
    N_CLASSES: int = 1 + 18

    MODEL: str = 'EfficientNetB0'
    LOSS: str = 'CategoricalFocalLoss'
    loss_params: dict = field(default_factory=dict)
    HEIGHT: int = 360
    WIDTH: int = 640
    SKIP_FRAME: int = 30

    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LR: float = 0.0001
    OPTIMIZER: str = 'AdamW'
    optim_params: dict = field(default_factory=dict)
    SCHEDULER: str = 'ReduceLROnPlateau'  # 'ReduceLROnPlateau' or 'CosineAnnealingLR'
    scheduler_params: dict = field(default_factory=dict)
    CLASS_WEIGHTS: Optional[List[float]] = None

    # augmentation
    TRAINING_AUGORATION: list = field(default_factory=list)
    VALIDATION_AUGORATION: list = field(default_factory=list)

    def __post_init__(self):
        '''list, dictはこちらで追記する'''
        # class mapping
        self.CLASSES = [
            'Toric Marking', 'Implant Ejection', 'Incision', 'Viscodilatation', 'Capsulorhexis', 'Hydrodissetion',
            'Nucleus Breaking', 'Phacoemulsification', 'Vitrectomy', 'Irrigation/Aspiration', 'Preparing Implant',
            'Manual Aspiration', 'Implantation', 'Positioning', 'OVD Aspiration', 'Suturing', 'Sealing Control',
            'Wound Hydratation'
        ]

        # optimizer params
        self.optim_params = {
            'weight_decay': 1e-5,
        }

        # learning rate scheduler parameters
        # cosine decay
        # self.scheduler_params = {
        #     'T_max': self.EPOCHS,
        #     'eta_min': self.LR * 0.1,
        # }
        # reduce on plateau
        self.scheduler_params = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'verbose': True,
        }

        self.TRAINING_AUGORATION = [

        ]

        self.VALIDATION_AUGORATION = [

        ]

        self.TRAINING_DIRS = [
            'train/01',
            # 'train/02',
            # 'train/03',
            # 'train/04',
            # 'train/05',
            # 'train/06',
            # 'train/07',
            # 'train/08',
            # 'train/09',
            # 'train/10',
            # 'train/11',
            # 'train/12',
            # 'train/13',
            # 'train/14',
            # 'train/15',
            # 'train/16',
            # 'train/17',
            # 'train/18',
            # 'train/19',
        ]

        self.VALIDATION_DIRS = [
            'train/20',
            # 'train/21',
            # 'train/22',
            # 'train/23',
            # 'train/24',
            # 'train/25',
        ]


if __name__ == '__main__':
    from pprint import pprint
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = Config()

    # Set random seed
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)

    # make save directory
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    # save train params as yaml
    with open(os.path.join(config.RESULT_DIR, 'parameters.yml'), 'w') as fw:
        pprint(asdict(config))  # show config
        fw.write(yaml.dump(asdict(config)))
    # fit
    main(config)
