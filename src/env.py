import os
import numpy as np
import pandas as pd
import gym


class ProcedureMaze(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, csv_file):
        super().__init__()
        self.map = self._generate_map(csv_file)
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(19)  # classes: 18 + 1
        self.observation_space = gym.spaces.Box(2)  # (frame, task)
        self.reward_range = [-1., 100.]
        self._reset()

    def _reset(self):
        """necessary method
        """
        self.done = False
        self.steps = 0

    def _step(self, action):
        """necessary method
        1ステップ進める処理
        戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        """
        next_pos = action

        if self._is_movable(next_pos):
            self.pos = next_pos
            moved = True
        else:
            moved = False

        observation = self._observe()
        reward = self._get_reward(self.pos, moved)
        self.done = self._is_done()
        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        """necessary method
        """

    def _get_reward(self, state, moved):
        """報酬を計算
        """
        if moved and (self.goal == state).all():
            # ゴールにたどり着くと 100 ポイント
            return 100
        else:
            # 1ステップごとに-1ポイント
            # (できるだけ短いステップでゴールにたどり着きたい)
            return -1

    def _is_done(self):
        """終了判定
        """
        if (self.pos == self.goal).all():
            return True
        elif self.steps > self.MAX_STEPS:
            return True
        else:
            return False

    def _observe(self):
        # マップに現在の位置を重ねて返す
        observation = self.MAP.copy()
        observation[tuple(self.pos)] = self.FIELD_TYPES.index('Y')
        return observation

    def _generate_map(self, csv_file):
        df = pd.read_csv(csv_file)
        onehot = np.eye(19)[df["Steps"]]  # 19 classes
        return onehot.T

    def _find_pos(self, field_type):
        return np.array(list(zip(*np.where(
            self.MAP == self.FIELD_TYPES.index(field_type)
        ))))

    def _is_movable(self, pos):
        # マップの中にいるか、歩けない場所にいないか
        return (
            0 <= pos[0] < self.MAP.shape[0]
            and 0 <= pos[1] < self.MAP.shape[1]
            and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'A'
        )
