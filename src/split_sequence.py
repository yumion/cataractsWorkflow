import os
from pprint import pprint
from glob import glob
from tqdm import tqdm
import cv2


def movie2frames(video_path, save_dir, skip_frame=1):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    start, end = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Number of frames:', end)

    for f in tqdm(range(start, end, skip_frame)):
        # f番目のフレームにセット
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()

        if ret is False:
            print("can't read frame id: ", f + 1)
            continue

        cv2.imwrite(os.path.join(save_dir, f'{f+1:0=5}.png'), frame)

    cap.release()


if __name__ == '__main__':
    # pathを設定する
    target_dir = '/data1/github/MICCAI2020/cataractsWorkflow/data/train'

    video_dirs = sorted(glob(os.path.join(target_dir, '[0-9][0-9]')))
    print('total videos: ', len(video_dirs))
    pprint(video_dirs)

    for movie_num, path in enumerate(video_dirs):
        print('movie: ', movie_num, path)
        movie2frames(
            video_path=glob(os.path.join(path, '*.mp4'))[0],
            save_dir=os.path.join(path, 'frame'),
            skip_frame=1,
        )
