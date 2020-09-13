from glob import glob
import csv
import os
import shutil
import cv2


def get_start2end_id(csv_file, save_path):
    os.makedirs(save_path, exist_ok=True)

    video_file = csv_file.replace(".csv", ".mp4")
    video = cv2.VideoCapture(video_file)
    last_frame_id = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    with open(csv_file, "r", encoding="utf8", errors="ignore") as fr:
        reader = csv.reader(fr)
        next(reader)  # skip header

        start_frames = []
        end_frames = []
        class_ids = []

        previous_class_id = -1
        first_frame = True

        for row in reader:
            frame_id, class_id = int(row[0]), int(row[1])

            # 前フレームと同じclass_idなら飛ばす
            if class_id == previous_class_id:
                continue

            previous_class_id = class_id  # 1つ前のclass_idを保持
            # 工程の始まるフレーム番号とその工程（class_id）を記録する
            start_frames.append(frame_id)
            class_ids.append(class_id)
            # 一番はじめのフレームではend_frameを記録しない
            if first_frame:
                first_frame = False
                continue
            # class_idの変わった1つ前をend_frameにする
            end_frames.append(frame_id - 1)
        # first_frame フラグのためend_framesが周回遅れているので、最後に付け足す
        end_frames.append(last_frame_id)
        # zipで固める
        annotations = list(zip(start_frames, end_frames, class_ids))

        # csvにdumpする
        with open(os.path.join(save_path, "start_end_id.csv"), "w") as fw:
            writer = csv.writer(fw)
            # header = ["start_frame_id", "end_frame_id", "class_id"]
            # writer.writerow(header)
            writer.writerows(annotations)


if __name__ == "__main__":
    target_dir = "/mnt/cloudy_z/input/cataractsWorkflow/train"
    save_dir = "/mnt/cloudy_z/src/atsushi/farmer/example/data/cataractsWorkflow/train"

    for video_dir in sorted(glob(os.path.join(target_dir, "[0-9][0-9]"))):
        csv_file = glob(os.path.join(video_dir, "*.csv"))[0]
        print(csv_file)

        # videoの各フォルダ名のまま、親ディレクトリを変更して保存する場合
        if target_dir != save_dir:
            subdirname = os.path.basename(video_dir)  # ファイル直上のフォルダ名
            save_path = os.path.join(save_dir, subdirname)
            os.makedirs(save_path, exist_ok=True)

            video_file = csv_file.replace(".csv", ".mp4")
            # shutil.copy(video_file, save_path)
        else:
            save_path = video_dir

        get_start2end_id(csv_file, save_path)
