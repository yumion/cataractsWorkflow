import os
import numpy as np
import csv
import cv2
from glob import glob
from torch.utils.data import Dataset as BaseDataset


class ClassificationVideoDataset(BaseDataset):

    CLASSES = ['Idle',
               'Toric Marking', 'Implant Ejection', 'Incision', 'Viscodilatation', 'Capsulorhexis', 'Hydrodissetion',
               'Nucleus Breaking', 'Phacoemulsification', 'Vitrectomy', 'Irrigation/Aspiration', 'Preparing Implant',
               'Manual Aspiration', 'Implantation', 'Positioning', 'OVD Aspiration', 'Suturing', 'Sealing Control',
               'Wound Hydratation']

    def __init__(
            self,
            video_dirs,
            classes,
            augmentation=None,
            preprocessing=None,
            skip_frame=1,
    ):
        # read all videos as a path
        self.annotations = self._get_annotation_set(video_dirs, skip_frame)
        # convert str names to class values(keep orders)
        self.class_values = [self.CLASSES.index(cls_name) for cls_name in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        video_path, frame_id, class_id = self.annotations[i]
        # open video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        # switch BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cast to onehot
        # classesで渡したclassのみ学習させる
        label = [(class_id == v) for v in self.class_values]
        label = np.concatenate([[0], label]).astype(np.float32)  # add others class(as mentioned `Idle`)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image, label

    def __len__(self):
        return len(self.annotations)

    def _get_annotation_set(self, video_dirs, skip_frame=1):
        """merge all videos into list
        """
        annotations = []
        for video_dir in video_dirs:
            # read movie and annotation csv, respectively
            video_path = glob(os.path.join(video_dir, '*.mp4'))[0]
            csv_path = glob(os.path.join(video_dir, '*.csv'))[0]
            # merge all videos as `[video_path, frame_id, class_id]`
            with open(csv_path) as fr:
                reader = csv.reader(fr)
                next(reader)  # skip header
                for row in reader:
                    # cast `str` to `int`
                    frame_id = int(row[0]) - 1  # cv2では0番目から始まるのでずらす
                    class_id = int(row[1])
                    # if set skip_frame, decrease FPS
                    if skip_frame == 0 or frame_id % skip_frame == 0:
                        annotations.append(
                            [video_path, frame_id, class_id]
                        )
        return annotations


class VideoPredictDataset(BaseDataset):

    def __init__(
            self,
            video_dir,
            augmentation=None,
            preprocessing=None,
            skip_frame=1,
    ):
        # open video capture
        self.video_path = glob(os.path.join(video_dir, '*.mp4'))[0]
        self.video = cv2.VideoCapture(self.video_path)
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        # select frames using test
        self.frame_ids = self._get_frame_ids(self.video, skip_frame)

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.skip_frame = skip_frame

    def __getitem__(self, i):

        # set reading frame on video
        frame_id = self.frame_ids[i]
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.video.read()

        # finish read frame, release capture
        if ret is False:
            self.video.release()
            return

        # switch BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return frame_id, image

    def __len__(self):
        return len(self.frame_ids)

    def _get_frame_ids(self, cap, skip_frame=1):
        """select frames for test
        """
        frame_ids = [
            frame_id for frame_id in range(
                0, self.num_frames,  # start frame # is `0`
                skip_frame  # if skip_frame == 1, use all frames(so, just `int(cap.get(cv2.CAP_PROP_FRAME_COUNT))`)
            )
        ]
        return frame_ids
