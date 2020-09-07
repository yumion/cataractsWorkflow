import os
from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional
import yaml
import csv
from tqdm import tqdm
import numpy as np
import torch
import albumentations as albu

from dataset import VideoPredictDataset
from utils import get_preprocessing, normalization
from augmentations import get_augmentation_wrapper


def main(config):
    # load best saved checkpoint
    best_model = torch.load(config.TRAINED_MODEL)

    # send to cuda and change mode `eval`
    best_model = best_model.to(config.DEVICE)
    best_model.eval()

    # create Dataset and DataLoader
    preprocessing_fn = normalization

    # set augmentation
    test_augmentation = get_augmentation_wrapper(config.TEST_AUGORATION)

    # predict each videos
    for test_dir in config.TEST_DIRS:
        # set dataset
        test_dataset = VideoPredictDataset(
            os.path.join(config.DATA_DIR, test_dir),
            augmentation=test_augmentation(height=config.HEIGHT, width=config.WIDTH),
            preprocessing=get_preprocessing(preprocessing_fn),
            skip_frame=config.SKIP_FRAME,
        )
        video_name = os.path.basename(test_dataset.video_path)
        print('test on', video_name)
        print(f'number of frame: {len(test_dataset)} / {test_dataset.num_frames}')

        # predict per frame
        results = []
        for frame_id, image in tqdm(test_dataset):
            # cast to tensor
            x_tensor = torch.from_numpy(image[np.newaxis]).to(config.DEVICE)  # (1,C,H,W)
            # inference
            with torch.no_grad():
                prediction = best_model.forward(x_tensor)  # (1,C)
                # cast to ndarray
                prediction = prediction.cpu().numpy().round()
                # cast from onehot to class_id
                predicted_label = np.argmax(prediction, axis=1).squeeze()  # (1,)

            # if apply `skip_frame`, padding previous label
            for i in range(config.SKIP_FRAME):
                # if last frame of video is over, stop padding
                if frame_id + 1 + i > test_dataset.num_frames:
                    break
                results.append([frame_id + 1 + i, predicted_label])  # 答えのframe_idが1から始まるのでずらす

        # save prediction as csv
        save_dir = os.path.join(config.RESULT_DIR, test_dir)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, video_name.replace('.mp4', '.csv')), 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(['Frame', 'Steps'])  # writer header
            writer.writerows(results)


@dataclass
class Config:
    DEVICE: str = 'cuda'

    DATA_DIR: str = '/data1/github/MICCAI2020/cataractsWorkflow/data'
    TEST_DIRS: list = field(default_factory=list)

    RESULT_DIR: str = "/data1/github/MICCAI2020/cataractsWorkflow/result/cnn_only/efficientnetb7/predict"
    TRAINED_MODEL: str = "/data1/github/MICCAI2020/cataractsWorkflow/result/cnn_only/efficientnetb7/best_model.pth"

    # input data config
    HEIGHT: int = 360
    WIDTH: int = 640
    SKIP_FRAME: int = 300
    # augmentation
    TEST_AUGORATION: list = field(default_factory=list)

    def __post_init__(self):
        '''list, dictはこちらで追記する'''
        self.TEST_AUGORATION = [

        ]

        self.TEST_DIRS = [
            'train/20',
            # 'train/21',
            # 'train/22',
            # 'train/23',
            # 'train/24',
            # 'train/25',
        ]


if __name__ == '__main__':
    from pprint import pprint
    # for weights download
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # 0:TITAN V,1:Quadro RTX8000, 2: TITAN RTX
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = Config()

    # make save directory
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    # save train params as yaml
    with open(os.path.join(config.RESULT_DIR, 'test_params.yml'), 'w') as fw:
        pprint(asdict(config))  # show config
        fw.write(yaml.dump(asdict(config)))
    # test
    main(config)
