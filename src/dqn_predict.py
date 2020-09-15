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
import matplotlib.pyplot as plt

from env import ProcedureMaze


# Q関数の定義


class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))
        return y

# リプレイバッファの定義


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
    def __init__(self, num_state, num_action, gamma=0.99, lr=0.001, batch_size=32, memory_size=50000):
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma  # 割引率
        self.batch_size = batch_size  # Q関数の更新に用いる遷移の数
        self.qnet = QNetwork(num_state, num_action).to('cuda')
        self.target_qnet = copy.deepcopy(self.qnet)  # ターゲットネットワーク
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)

        self.qnet = self.qnet.to('cuda')
        self.target_qnet = self.target_qnet.to('cuda')

    # Q関数を更新
    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.qnet(torch.tensor(batch["states"], dtype=torch.float).to('cuda'))
        targetq = copy.deepcopy(q.data)
        # maxQの計算
        maxq = torch.max(self.target_qnet(torch.tensor(batch["next_states"], dtype=torch.float).to('cuda')), dim=1).values
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

    # Q値が最大の行動を選択
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to('cuda')
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
        class_weights = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0.9989, 0., 0.0011, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0.9986, 0., 0.0014, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0.9984, 0.0014, 0.0001, 0., 0., 0., 0.0001, 0., 0., 0., 0.0001, 0., 0., 0., 0., 0.],
                                      [0., 0., 0.0003, 0.0002, 0.9954, 0.0019, 0., 0., 0., 0.0001, 0., 0.001, 0., 0.0011, 0.0001, 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0.9992, 0.0008, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0.9976, 0.0024, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0.9992, 0.0008, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0.9995, 0., 0.0005, 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0.0001, 0.0001, 0., 0., 0., 0., 0.9995, 0., 0., 0.0004, 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0.0005, 0., 0., 0., 0., 0., 0.9995, 0., 0., 0., 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.999, 0., 0.001, 0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0.0001, 0., 0., 0., 0., 0.0004, 0., 0., 0.9995, 0., 0., 0., 0., 0.0001, 0.],
                                      [0., 0., 0., 0.0002, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9972, 0.0026, 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9987, 0.0009, 0., 0.0003, 0.0001],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9992, 0., 0., 0.0007],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0001, 0., 0.9995, 0., 0.0004],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0014, 0., 0.0002, 0.9984, 0.],
                                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0001, 0., 0.9999]],
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


if __name__ == '__main__':
    # Set random seed
    np.random.seed(124)
    torch.manual_seed(124)
    torch.cuda.manual_seed(124)

    # 各種設定
    csv_files = [
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/01/train01.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/02/train02.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/03/train03.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/04/train04.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/05/train05.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/06/train06.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/07/train07.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/08/train08.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/09/train09.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/10/train10.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/11/train11.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/12/train12.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/13/train13.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/14/train14.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/15/train15.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/16/train16.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/17/train17.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/18/train18.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/19/train19.csv',
        # '/mnt/cloudy_z/input/cataractsWorkflow/train/20/train20.csv',
        '/mnt/cloudy_z/input/cataractsWorkflow/train/21/train21.csv',
        '/mnt/cloudy_z/input/cataractsWorkflow/train/22/train22.csv',
        '/mnt/cloudy_z/input/cataractsWorkflow/train/23/train23.csv',
        '/mnt/cloudy_z/input/cataractsWorkflow/train/24/train24.csv',
        '/mnt/cloudy_z/input/cataractsWorkflow/train/25/train25.csv',
    ]
    skip_frame = 1
    # モデルのパスとを変えてください
    model_path = '/mnt/cloudy_z/src/atsushi/cataractsWorkflow/result/dqn/weighted_trains_all/model/checkpoint_ep330.pth'
    result_dir = 'result/dqn/test'

    # ログ
    os.makedirs(result_dir, exist_ok=True)

    agent = DqnAgent(num_state=2, num_action=19, memory_size=5000)
    agent.load_model(model_path)
    # 最終的に得られた方策のテスト（可視化）
    for csv_file in csv_files:
        env = ProcedureMaze(csv_file=csv_file, skip_frame=skip_frame)
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
            writer.writerows(np.asarray(list(enumerate(np.array(frames)[:, 0], 1)), dtype=np.int))

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
