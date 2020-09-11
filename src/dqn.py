import os
import numpy as np
import copy
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from env import ProcedureMaze


"""## 2.価値に基づく手法 (Value-based Methods)
価値に基づく手法では，価値に基づいて行動を決定する方策を利用します．
- 方策$\pi$として，小さい確率$\epsilon$で一様な確率でランダムな行動を選択し，それ以外は最もQ値（の推定値）が最も高い行動を選択する，**ε-greedy方策**が用いられることが多いです．
  - ランダムな行動を選択することで，探索を促進するために利用されます．

今回は，価値に基づく代表的な手法として，**SARSA**と**Q-learning**を扱います．
- どちらも以下の再帰的な更新により，Q関数を推定します，$$Q^{\pi}(s_t,a_t) \leftarrow Q^{\pi}(s_t,a_t)+\alpha \delta_t$$$$\delta_t = y_t - Q^{\pi}(s_t,a_t)$$
  - これは，Q値を目標値$y_t$に向かって更新する操作になっています．
  - $\alpha$：**学習率**（ステップサイズ）
  - $\delta_t$：**TD誤差** (temporal difference error)
  - $y_t$：**TDターゲット**
SARSAとQ-learningでTDターゲット$y_t$の求め方が異なります．$$y^{SARSA}_t = r_{t+1}+\gamma Q^{\pi}(s_{t+1},a_{t+1})$$$$y^{Q-learning}_t = r_{t+1}+\gamma \max_{a'}Q^{\pi}(s_{t+1},a')$$
  - $\gamma$：**割引率**


# 2.2.1 Deep Q-Network（DQN）
DQN（Deep Q-Network）は，Q関数をNNによって近似した手法です．Q-learningのTD誤差は，$$\delta_t = r_{t+1}+\gamma \max_{a'}Q^{\pi}(s_{t+1},a') - Q^{\pi}(s_t,a_t)$$であるので，$r_{t+1}+\gamma \max_{a'}Q^{\pi}(s_{t+1},a')$と$Q^{\pi}(s_t,a_t)$の差の最小化をすればいいことがわかります．
- 今回の実装ではMSEを用いています．

**特徴**
- **経験リプレイ**（experience replay）の利用
  - **リプレイバッファ**（replay buffer）にこれまでの状態遷移を記録しておき，そこからサンプルすることでQ関数のミニバッチ学習をします．
- **固定したターゲットネットワーク**（fixed target Q-network）の利用
  - 学習する対象のQ関数$Q^{\pi}(s_t,a_t)$も，更新によって近づける目標値$r_{t+1}+\gamma \max_{a'}Q^{\pi}(s_{t+1},a')$もどちらも同じパラメータを持つQ関数を利用しています．そのため，そのまま勾配法による最適化を行うと，元のQ値と目標値の両方が更新されてしまうことになります．
  - これを避けるために，目標値$r_{t+1}+\gamma \max_{a'}Q^{\pi}(s_{t+1},a')$は固定した上でQ関数の最適化を行います．
    - 実装上は，目標値側のQ関数の勾配の情報を削除して数値として扱います．

**論文**
- [Human-level control through deep reinforcement learning （Nature版）](https://www.nature.com/articles/nature14236)
- [Playing Atari with Deep Reinforcement Learning （NIPS2013 workshop版）](https://arxiv.org/abs/1312.5602)


**補足説明**
- 深層強化学習には様々な実装の仕方がありますが，今回の講義では以下のモジュールに分けて実装しています．
  - 1つ目のセル：関数近似のためのニューラルネットワーク（とリプレイバッファ）
  - 2つ目のセル：エージェントの定義
  - 3つ目のセル：実際に環境と相互作用して学習を実行する部分
  - (4つ目のセル：学習したエージェントの結果の可視化）
"""

# Q関数の定義


class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
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
        self.qnet = QNetwork(num_state, num_action)
        self.target_qnet = copy.deepcopy(self.qnet)  # ターゲットネットワーク
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)

    # Q関数を更新
    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.qnet(torch.tensor(batch["states"], dtype=torch.float))
        targetq = copy.deepcopy(q.data.numpy())
        # maxQの計算
        maxq = torch.max(self.target_qnet(torch.tensor(batch["next_states"], dtype=torch.float)), dim=1).values
        # Q値が最大の行動だけQ値を更新（最大ではない行動のQ値はqとの2乗誤差が0になる）
        for i in range(self.batch_size):
            # 終端状態の場合はmaxQを0にしておくと学習が安定します（ヒント：maxq[i] * (not batch["dones"][i])）
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + self.gamma * maxq[i] * (not batch["dones"][i])
        self.optimizer.zero_grad()
        # lossとしてMSEを利用
        loss = nn.MSELoss()(q, torch.tensor(targetq))
        loss.backward()
        self.optimizer.step()
        # ターゲットネットワークのパラメータを更新
        self.target_qnet = copy.deepcopy(self.qnet)

    # Q値が最大の行動を選択
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action = torch.argmax(self.qnet(state_tensor).data).item()
        return action

    # ε-greedyに行動を選択
    def get_action(self, state, episode):
        epsilon = 0.7 * (1 / (episode + 1))  # ここでは0.5から減衰していくようなεを設定
        if epsilon <= np.random.uniform(0, 1):
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

    # 各種設定
    csv_file = '/data1/github/MICCAI2020/cataractsWorkflow/data/train/01/train01.csv'
    result_dir = 'result/dqn/train01'
    num_episode = 100  # 学習エピソード数
    memory_size = 100000  # replay bufferの大きさ
    initial_memory_size = 1000  # 最初に貯めるランダムな遷移の数

    # ログ
    os.makedirs(result_dir, exist_ok=True)
    episode_rewards = []
    num_average_epidodes = 10

    env = ProcedureMaze(csv_file=csv_file, skip_frame=1)
    agent = DqnAgent(env.observation_space.shape[0], env.action_space.n, memory_size=memory_size)

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

    for episode in tqdm(range(num_episode)):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, episode)  # 行動を選択
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
            agent.update_q()  # Q関数を更新
            state = next_state

        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_reward))
            agent.save_model(save_path=os.path.join(result_dir, 'checkpoint.pth'))

    # 累積報酬の移動平均を表示
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)), moving_average)
    plt.title('DQN: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    # plt.show()
    plt.savefig(os.path.join(result_dir, 'train_rewards.png'))

    env.close()

    # 最終的に得られた方策のテスト（可視化）
    for episode in range(1):
        env = ProcedureMaze(csv_file=csv_file, skip_frame=1)
        frames = []

        state = env.reset()
        frames.append(env.render(mode='human'))
        done = False
        while not done:
            action = agent.get_greedy_action(state)
            state, reward, done, _ = env.step(action)
            frames.append(env.render(mode='human'))

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
        plt.savefig(os.path.join(result_dir, f'test_pred{episode}.png'))

        env.close()
