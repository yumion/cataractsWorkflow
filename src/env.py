import numpy as np
import pandas as pd
import gym


class ProcedureMaze(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, csv_file, skip_frame=1):
        """`action_space`, `observation_space`, `reward_range` are necessary
        Observation:
        Type: Box(2)
        Num     Observation     Min                  Max
        0       frame_id          1      number of frame
        1       task              0                   18
        Actions:
        Type: Discrete(19)
        Num   Action
        0     Idle
        1~18  task
        """
        super().__init__()
        # 予測結果から迷路を生成
        self.maze = self._generate_maze(csv_file, skip_frame)  # (0, 0)からスタート

        self.action_space = gym.spaces.Discrete(19)  # classes: 18 + 1
        self.observation_space = gym.spaces.Box(
            low=np.array([1, 0], dtype=np.float32),
            high=np.array([self.maze.shape[0], 19], dtype=np.float32),
            shape=(2, )
        )  # [frame, task]
        self.reward_range = [-1., 100.]
        # 環境を初期化
        self.reset()

    def reset(self):
        """necessary method
        """
        self.steps = 0
        self.pos = [1, 0]  # はじめはframe_id=1,task=0から始まる
        self.done = False
        return self._observe()

    def step(self, action):
        """necessary method
        actionを行った処理（1回ごと）
        return: observation(state), reward, done(episode終了判定), info(追加の情報の辞書)
        """
        # x座標は常に進める
        # actionがそのままy座標となる
        self.pos = [self.pos[0] + 1, action]

        next_pos = self._observe()
        reward = self._get_reward(next_pos)
        self.done = self._is_done(next_pos)

        return next_pos, reward, self.done, {}

    def _get_reward(self, pos):
        """報酬を計算
        TODO:報酬をいくつに設定するか？
        ゴールだけが正しいことに意味はないので、ゴールの報酬は特に設定しない
        """
        current_task = np.argmax(self.maze[pos[0] - 1])
        if pos[1] == current_task == 0:
            # 予測のtaskと同じでもIdle(0)だと少ない報酬を与える
            return 0
        elif pos[1] == current_task:
            # 予測のtaskと同じ(Idle以外)だと大きな報酬を与える
            return 1
        else:
            # 予測のtaskと異なるとペナルティを与える
            return -1

    def _is_done(self, pos):
        """終了判定
        x軸方向に進めなくなったらTrueを返す
        """
        return self.maze.shape[0] == pos[0]

    def _observe(self):
        """次のstateを返す
        移動先の座標を観測とする
        """
        return self.pos

    def _generate_maze(self, csv_file, skip_frame=1):
        df = pd.read_csv(csv_file)
        onehot = np.eye(19)[df["Steps"]]  # 19 classes
        return onehot[::skip_frame]

    def render(self, mode='human', close=False):
        """necessary method
        """
        if mode == 'rgb_array':
            output = self.maze.copy()
            output[self.pos[0] - 1, self.pos[1]] = 2  # 現在地を2とする
            return output

        elif mode == 'human':
            target = self.maze.copy()
            target = np.argmax(target, axis=-1)  # cast to class_id from onehot
            # [class_id(pred), class_id(target)]
            output = [self.pos[1], target[self.pos[0] - 1]]
            return output
