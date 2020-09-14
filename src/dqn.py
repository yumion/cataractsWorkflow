import os
import numpy as np
import copy
from collections import deque
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import matplotlib.pyplot as plt

import albumentations as albu
from env import ProcedureMaze
from dataset import VideoPredictDataset
from augmentations import get_augmentation_wrapper
from utils import normalization


class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        # h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))
        return y


class FeatureExtractor:
    def __init__(self, feature_model_path=None):
        super(FeatureExtractor, self).__init__()
        if feature_model_path is not None:
            base_model = tf.keras.models.load_model(feature_model_path)
            # GAPした後まで
            # input.shape=(None, 360, 640, 3)
            # output.shape=(None, 2048)
            self.feature_model = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=base_model.get_layer(index=-2).output
            )
        else:
            base_model = tf.keras.applications.Xception(include_top=False)
            x = base_model.outputs
            x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
            self.feature_model = tf.keras.Model(inputs=base_model.input, outputs=x)

    def __call__(self, image_batch):
        feature_vec = self.feature_model.predict(image_batch)
        return feature_vec


class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque([], maxlen=memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states = np.array([self.memory[index]['state'] for index in batch_indexes])
        next_states = np.array([self.memory[index]['next_state'] for index in batch_indexes])
        rewards = np.array([self.memory[index]['reward'] for index in batch_indexes])
        actions = np.array([self.memory[index]['action'] for index in batch_indexes])
        dones = np.array([self.memory[index]['done'] for index in batch_indexes])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions, 'dones': dones}


class DqnAgent:
    def __init__(self, num_state, num_action, dataset, gamma=0.99, lr=0.001, batch_size=32, memory_size=50000):
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma  # 割引率
        self.batch_size = batch_size  # Q関数の更新に用いる遷移の数
        self.feature_extractor = FeatureExtractor('/data1/github/MICCAI2020/cataractsWorkflow/result/cnn_only/tf-xception-skipframe=1_trial2/model/best_model.h5')
        self.qnet = QNetwork(num_state, num_action, hidden_size=2048).to('cuda')
        self.target_qnet = copy.deepcopy(self.qnet).to('cuda')  # ターゲットネットワーク
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)
        self.dataset = dataset

    # Q関数を更新
    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        # q = self.qnet(torch.tensor(batch["states"], dtype=torch.float).to('cuda'))

        x_tensor = self.feature_extract(batch["states"])
        q = self.qnet(x_tensor.to('cuda'))

        targetq = copy.deepcopy(q.data)
        # maxQの計算
        # maxq = torch.max(self.target_qnet(torch.tensor(batch["next_states"], dtype=torch.float).to('cuda')), dim=1).values
        maxq = torch.max(self.target_qnet(self.feature_extract(batch["next_states"]).to('cuda')), dim=1).values
        # Q値が最大の行動だけQ値を更新（最大ではない行動のQ値はqとの2乗誤差が0になる）
        for i in range(self.batch_size):
            # 終端状態の場合はmaxQを0にしておくと学習が安定します（ヒント：maxq[i] * (not batch["dones"][i])）
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + self.gamma * maxq[i] * (not batch["dones"][i])
        self.optimizer.zero_grad()
        # lossとしてMSEを利用
        mse_loss = nn.MSELoss().to('cuda')
        loss = mse_loss(q, targetq)
        loss.backward()
        self.optimizer.step()
        # ターゲットネットワークのパラメータを更新
        self.target_qnet = copy.deepcopy(self.qnet)

        return loss

    def feature_extract(self, state):
        frames = np.array([self.dataset[i][1] for i in state[:, 0]])
        feature_vec = self.feature_extractor(frames)
        frame_ids = np.array([self.dataset[i][0] for i in state[:, 0]])
        # cast to torch.tensor
        x = torch.cat([torch.from_numpy(feature_vec),
                       torch.tensor(frame_ids, dtype=torch.float).unsqueeze(-1)],
                      dim=1)
        return x

    # Q値が最大の行動を選択
    def get_greedy_action(self, state):
        # state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to('cuda')
        state_tensor = self.feature_extract(np.array([state[0]])[np.newaxis]).to('cuda')  # state[0]: frame_id, state[1]: action
        action = torch.argmax(self.qnet(state_tensor).data).item()
        return action

    # 工程の遷移頻度割合でQ値に重み付けしてargmaxする
    def get_weighted_action(self, state, previous_action, offset=1.):
        # 連続する工程は無視
        # class_weights = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0.86666667, 0.03333333, 0., 0., 0., 0.03333333, 0., 0., 0., 0.06666667, 0., 0., 0., 0., 0.],
        #                               [0., 0., 0.07017544, 0.03508772, 0., 0.42105263, 0., 0., 0., 0.01754386, 0., 0.21052632, 0., 0.22807018, 0.01754386, 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0.14285714, 0.14285714, 0., 0., 0., 0., 0., 0., 0., 0.71428571, 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0.96, 0., 0., 0., 0., 0., 0., 0., 0.04, 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0.16666667, 0., 0., 0., 0., 0.66666667, 0., 0., 0., 0., 0., 0., 0., 0.16666667, 0.],
        #                               [0., 0., 0., 0.07407407, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.92592593, 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.02777778, 0., 0., 0., 0., 0.02777778, 0.66666667, 0.02777778, 0.19444444, 0.05555556],
        #                               [0., 0., 0., 0., 0.04, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.04, 0.04, 0.04, 0., 0.84],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.2, 0., 0., 0., 0.8],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.875, 0., 0.125, 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.66666667, 0.33333333, 0.]],
        #                              dtype=torch.float).to('cuda')
        # 連続する工程も考慮（工程0は除く）
        # class_weights = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0.9989, 0., 0.0011, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0.9986, 0., 0.0014, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0.9984, 0.0014, 0.0001, 0., 0., 0., 0.0001, 0., 0., 0., 0.0001, 0., 0., 0., 0., 0.],
        #                               [0., 0., 0.0003, 0.0002, 0.9954, 0.0019, 0., 0., 0., 0.0001, 0., 0.001, 0., 0.0011, 0.0001, 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0.9992, 0.0008, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0.9976, 0.0024, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0.9992, 0.0008, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0.9995, 0., 0.0005, 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0.0001, 0.0001, 0., 0., 0., 0., 0.9995, 0., 0., 0.0004, 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0.0005, 0., 0., 0., 0., 0., 0.9995, 0., 0., 0., 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.999, 0., 0.001, 0., 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0.0001, 0., 0., 0., 0., 0.0004, 0., 0., 0.9995, 0., 0., 0., 0., 0.0001, 0.],
        #                               [0., 0., 0., 0.0002, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9972, 0.0026, 0., 0., 0., 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9987, 0.0009, 0., 0.0003, 0.0001],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9992, 0., 0., 0.0007],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0001, 0., 0.9995, 0., 0.0004],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0014, 0., 0.0002, 0.9984, 0.],
        #                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0001, 0., 0.9999]],
        #                              dtype=torch.float).to('cuda')
        # 工程0も含める
        class_weights = torch.tensor([[0.99725234, 0.00000747, 0.00002987, 0.00020906, 0.00042559, 0.00018666, 0.00018666, 0.00018666, 0.0001792, 0.00005227, 0.00018666, 0.0000896, 0.0000448, 0.00020159, 0.00026879,
                                       0.00018666, 0.00003733, 0.0000672, 0.00020159],
                                      [0.00111732, 0.99888268, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00141143, 0., 0.99858857, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00162593, 0., 0., 0.99837407, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00462512, 0., 0., 0., 0.99537488, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00082242, 0., 0., 0., 0., 0.99917758, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00239854, 0., 0., 0., 0., 0., 0.99760146, 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00075882, 0., 0., 0., 0., 0., 0., 0.99920956, 0.00003162, 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00050993, 0., 0., 0., 0., 0., 0., 0., 0.99949007, 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00050505, 0., 0., 0., 0., 0., 0., 0., 0., 0.99949495, 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00050956, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.99949044, 0., 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00098184, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.99901816, 0., 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.0005421, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9994579, 0., 0.,
                                       0., 0., 0., 0.],
                                      [0.00281955, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.99718045, 0.,
                                       0., 0., 0., 0.],
                                      [0.00132631, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.99867369,
                                       0., 0., 0., 0.],
                                      [0.00080391, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0.99919609, 0., 0., 0.],
                                      [0.00048412, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0.99951588, 0., 0.],
                                      [0.00176922, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0.99823078, 0.],
                                      [0.00076082, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0.99923918]],
                                     dtype=torch.float).to('cuda')

        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to('cuda')
        action = torch.argmax(self.qnet(state_tensor).data * (class_weights[previous_action] + offset)).item()
        return action

    # ε-greedyに行動を選択
    def get_action(self, state, episode, **kwargs):
        epsilon = 0.7 * (1 / (episode + 1))  # ここでは0.5から減衰していくようなεを設定
        if epsilon <= np.random.uniform(0, 1):
            if kwargs:
                action = self.get_weighted_action(state, **kwargs)
            else:
                action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action

    def save_model(self, save_path='last_model.pth'):
        torch.save(self.qnet, save_path)

    def load_model(self, model_path):
        self.qnet = torch.load(model_path)
        self.target_qnet = copy.deepcopy(self.qnet)


def get_preprocessing():
    _transform = [albu.Lambda(image=normalization)]
    return albu.Compose(_transform)


if __name__ == '__main__':
    # Set random seed
    np.random.seed(124)
    torch.manual_seed(124)
    torch.cuda.manual_seed(124)

    # 0:TITAN V,1:Quadro RTX8000, 2: TITAN RTX
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # GPUメモリ使用量を抑える
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth: ', tf.config.experimental.get_memory_growth(physical_devices[k]))

    # 各種設定
    csv_files = [
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/01/train01.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/02/train02.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/03/train03.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/04/train04.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/05/train05.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/06/train06.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/07/train07.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/08/train08.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/09/train09.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/10/train10.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/11/train11.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/12/train12.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/13/train13.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/14/train14.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/15/train15.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/16/train16.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/17/train17.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/18/train18.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/19/train19.csv',
        '/data1/github/MICCAI2020/cataractsWorkflow/data/train/20/train20.csv',
    ]

    result_dir = 'result/dqn/feature_extract_trains_all'
    os.makedirs(os.path.join(result_dir, 'model'), exist_ok=True)

    num_episode = 1000  # 学習エピソード数
    memory_size = 100000  # replay bufferの大きさ
    initial_memory_size = 1000  # 最初に貯めるランダムな遷移の数
    skip_frame = 600
    test_augmentation = get_augmentation_wrapper([])

    # ログ
    os.makedirs(result_dir, exist_ok=True)
    episode_rewards = []
    num_average_epidodes = 10

    maze_csv = np.random.choice(csv_files)
    env = ProcedureMaze(csv_file=maze_csv, skip_frame=skip_frame)
    dataset = VideoPredictDataset(
        os.path.dirname(maze_csv),
        augmentation=test_augmentation(height=360, width=640),
        preprocessing=get_preprocessing(),
        skip_frame=skip_frame,
    )
    agent = DqnAgent(num_state=2048 + 1,  # env.observation_space.shape[0]
                     num_action=env.action_space.n,
                     dataset=dataset,
                     memory_size=memory_size)

    # 最初にreplay bufferにランダムな行動をしたときのデータを入れる
    state = env.reset()
    for step in range(initial_memory_size):
        action = env.action_space.sample()  # ランダムに行動を選択
        next_state, reward, done, _ = env.step(action)
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'done': int(done)
        }
        agent.replay_buffer.append(transition)
        state = env.reset() if done else next_state
    env.close()

    for episode in tqdm(range(num_episode)):
        # episodeごとに迷路を変える
        maze_csv = np.random.choice(csv_files)
        print(maze_csv)
        env = ProcedureMaze(csv_file=maze_csv, skip_frame=skip_frame)
        # 動画も変える
        dataset = VideoPredictDataset(
            os.path.dirname(maze_csv),
            augmentation=test_augmentation(height=360, width=640),
            preprocessing=get_preprocessing(),
            skip_frame=skip_frame,
        )
        agent.dataset = dataset

        state = env.reset()
        previous_action = 0
        episode_reward = 0
        loss = 0
        done = False

        while not done:
            print(state)
            action = agent.get_action(
                state, episode,
                # previous_action=previous_action, offset=1.
            )  # 行動を選択
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            transition = {
                'state': state,
                'next_state': next_state,
                'reward': reward,
                'action': action,
                'done': int(done)
            }
            agent.replay_buffer.append(transition)
            loss += agent.update_q()  # Q関数を更新
            state = next_state
            previous_action = action

        episode_rewards.append(episode_reward)
        if episode % 1 == 0:
            print("Episode %d finished | Episode reward %f | Episode loss %f" % (episode, episode_reward, loss / 10))
            agent.save_model(save_path=os.path.join(result_dir, 'model', f'checkpoint_ep{episode}.pth'))

        env.close()

    agent.save_model(save_path=os.path.join(result_dir, 'model', 'last_model.pth'))

    # 累積報酬の移動平均を表示
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)), moving_average)
    plt.title('DQN: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    # plt.show()
    plt.savefig(os.path.join(result_dir, 'train_rewards.png'))

    # 最終的に得られた方策のテスト（可視化）
    for csv_file in csv_files:
        env = ProcedureMaze(csv_file=csv_file, skip_frame=skip_frame)
        # 動画も変える
        dataset = VideoPredictDataset(
            os.path.dirname(csv_file),
            augmentation=test_augmentation(height=360, width=640),
            preprocessing=get_preprocessing(),
            skip_frame=skip_frame,
        )
        agent.dataset = dataset
        frames = []

        state = env.reset()
        frames.append(env.render(mode='human'))
        done = False
        while not done:
            action = agent.get_greedy_action(state)
            state, reward, done, _ = env.step(action)
            frames.append(env.render(mode='human'))

        # 予測をcsvで保存
        with open(f'pred_{os.path.basename(csv_file)}', 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(['Frame', 'Steps'])
            writer.writerows(np.asarray(list(enumerate(np.array(frames[:, 0]), 1)), dtype=np.int))

        # 予測をグラフで保存
        preds = []
        targets = []
        for frame in frames[:]:
            preds.append(frame[0])
            targets.append(frame[1])

        plt.figure(figsize=(16, 6))
        plt.yticks(np.arange(0, 19, 1))
        plt.plot(preds)
        plt.plot(targets)
        # plt.show()
        plt.savefig(os.path.join(result_dir, f"test_{os.path.basename(csv_file).rstrip('.csv')}.png"))

        env.close()
