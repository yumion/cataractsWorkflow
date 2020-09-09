import os
import numpy as np
import pandas as pd
import gym


class ProcedureMaze(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, csv_file, skip_frame=1):
        """`action_space`, `observation_space`, `reward_range` are necessary
        """
        super().__init__()
        # 予測結果から迷路を生成
        self.maze = self._generate_maze(csv_file, skip_frame)

        self.action_space = gym.spaces.Discrete(19)  # classes: 18 + 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.maze.shape[0],
            shape=(2, )
        )  # [frame, task]
        self.reward_range = [-1., 1000.]
        # 環境を初期化
        self.reset()

    def reset(self):
        """necessary method
        """
        self.steps = 0
        self.pos = [1, 0]  # はじめはframe_id=1,task=0から始まる
        self.goal = [self.maze.shape[0], np.argmax(self.maze[-1])]  # ゴールはフレームの最後とそのtask_id
        self.done = False
        return self.pos

    def step(self, action):
        """necessary method
        actionを行った処理（1回ごと）
        return: observation(state), reward, done(episode終了判定), info(追加の情報の辞書)
        """
        # x座標は常に進める
        # actionがそのままy座標となる
        next_pos = [self.pos[0] + 1, action]

        if self._is_movable(next_pos):
            self.pos = next_pos
        else:
            self.done = True

        # observation = self._observe()
        observation = self.pos  # 移動先の座標を観測とする
        reward = self._get_reward(self.pos)
#         self.done = self._is_done()
        return observation, reward, self.done, {}

    def render(self, mode='human', close=False):
        """necessary method
        """
        if mode == 'rgb_array':
            output = self.maze.copy()
            output[self.pos[0], self.pos[1]] = 2  # 現在地を2とする
            return output

        elif mode == 'human':
            target = self.maze.copy()
            target = np.argmax(target, axis=-1)  # to class_id from onehot
            # [class_id(pred), class_id(target)]
            output = [self.pos[1], target[self.pos[0]]]
            return output

    def _get_reward(self, pos):
        """報酬を計算
        """
        if self.goal == pos:
            # ゴールにたどり着くと高い報酬を与える
            # TODO: ゴールの報酬をいくつに設定するか？
            return 1000
        elif self.pos[1] == np.argmax(self.maze[self.pos[0]]):
            # 予測のtaskと同じだと報酬を与える
            return 1
        else:
            # 予測のtaskと異なるとペナルティを与える
            return -1

    def _is_done(self):
        """終了判定
        TODO: これだとdoneがTrueにならない
        """
        if self.pos[0] == self.maze.shape[0]:
            return True
        else:
            return False

    def _observe(self):
        """手術画像を入力とするときに使用
        """
        pass

    def _generate_maze(self, csv_file, skip_frame=1):
        df = pd.read_csv(csv_file)
        onehot = np.eye(19)[df["Steps"]]  # 19 classes
        return onehot[::skip_frame]

    def _is_movable(self, pos):
        # x軸方向に進めなくなったらFalseを返す
        return self.goal[0] != pos[0]
