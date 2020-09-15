import os
from dataclasses import dataclass, field, asdict
import yaml
import csv
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import albumentations as albu

from dataset import VideoPredictDataset
from utils import normalization
from augmentations import get_augmentation_wrapper


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
    ]
    return albu.Compose(_transform)


def main(config):
    # load best saved checkpoint
    best_model = tf.keras.models.load_model(config.TRAINED_MODEL)

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
            x_batch = image[np.newaxis]  # (1,C,H,W)
            # inference
            prediction = best_model.predict(x_batch)  # (1,C)
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

    RESULT_DIR: str = "/data1/github/MICCAI2020/cataractsWorkflow/result/cnn_only/tf-xception-skipframe=1_trial2/predict"
    TRAINED_MODEL: str = "/data1/github/MICCAI2020/cataractsWorkflow/result/cnn_only/tf-xception-skipframe=1_trial2/model/best_model.h5"

    # input data config
    HEIGHT: int = 360
    WIDTH: int = 640
    SKIP_FRAME: int = 1
    # augmentation
    TEST_AUGORATION: list = field(default_factory=list)

    def __post_init__(self):
        '''list, dictはこちらで追記する'''
        self.TEST_AUGORATION = [

        ]

        self.TEST_DIRS = [
            # 'train/01',
            # 'train/02',
            # 'train/03',
            # 'train/04',
            # 'train/05',
            # 'train/06',
            # 'train/07',
            'train/08',
            'train/09',
            'train/10',
            'train/11',
            'train/12',
            'train/13',
            'train/14',
            'train/15',
            'train/16',
            'train/17',
            'train/18',
            'train/19',
            'train/20',
            # 'train/21',
            # 'train/22',
            # 'train/23',
            # 'train/24',
            # 'train/25',
            # 'validation/01',
            # 'validation/02',
            'validation/03',
            'validation/04',
            'validation/05',
        ]


if __name__ == '__main__':
    from pprint import pprint
    # for weights download
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # 0:TITAN V,1:Quadro RTX8000, 2: TITAN RTX
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # GPUメモリ使用量を抑える
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth: ', tf.config.experimental.get_memory_growth(physical_devices[k]))

    config = Config()

    # make save directory
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    # save train params as yaml
    with open(os.path.join(config.RESULT_DIR, 'test_params.yml'), 'w') as fw:
        pprint(asdict(config))  # show config
        fw.write(yaml.dump(asdict(config)))
    # test
    main(config)
