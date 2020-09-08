
# In[1]:

import numpy as np
import pandas as pds
import random
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from collections import deque
from keras import backend as K


# # Maze Class

# In[2]:

class Maze(object):
    def __init__(self, size=10, blocks_rate=0.1):
        self.size = size if size > 3 else 10
        self.blocks = int((size ** 2) * blocks_rate)
        self.s_list = []
        self.maze_list = []
        self.e_list = []

    def create_mid_lines(self, k):
        if k == 0:
            self.maze_list.append(self.s_list)
        elif k == self.size - 1:
            self.maze_list.append(self.e_list)
        else:
            tmp_list = []
            for l in range(0, self.size):
                if l == 0:
                    tmp_list.extend("#")
                elif l == self.size - 1:
                    tmp_list.extend("#")
                else:
                    a = random.randint(-1, 0)
                    tmp_list.extend([a])
            self.maze_list.append(tmp_list)

    def insert_blocks(self, k, s_r, e_r):
        b_y = random.randint(1, self.size - 2)
        b_x = random.randint(1, self.size - 2)
        if [b_y, b_x] == [1, s_r] or [b_y, b_x] == [self.size - 2, e_r]:
            k = k - 1
        else:
            self.maze_list[b_y][b_x] = "#"

    def generate_maze(self):
        s_r = random.randint(1, (self.size / 2) - 1)
        for i in range(0, self.size):
            if i == s_r:
                self.s_list.extend("S")
            else:
                self.s_list.extend("#")
        start_point = [0, s_r]

        e_r = random.randint((self.size / 2) + 1, self.size - 2)
        for j in range(0, self.size):
            if j == e_r:
                self.e_list.extend([50])
            else:
                self.e_list.extend("#")
        goal_point = [self.size - 1, e_r]

        for k in range(0, self.size):
            self.create_mid_lines(k)

        for k in range(self.blocks):
            self.insert_blocks(k, s_r, e_r)

        return self.maze_list, start_point, goal_point


# # Maze functions

# In[3]:

class Field(object):
    def __init__(self, maze, start_point, goal_point):
        self.maze = maze
        self.start_point = start_point
        self.goal_point = goal_point
        self.movable_vec = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def display(self, point=None):
        field_data = copy.deepcopy(self.maze)
        if not point is None:
            y, x = point
            field_data[y][x] = "@@"
        else:
            point = ""
        for line in field_data:
            print("\t" + "%3s " * len(line) % tuple(line))

    def get_actions(self, state):
        movables = []
        if state == self.start_point:
            y = state[0] + 1
            x = state[1]
            a = [[y, x]]
            return a
        else:
            for v in self.movable_vec:
                y = state[0] + v[0]
                x = state[1] + v[1]
                if not(0 < x < len(self.maze) and
                       0 <= y <= len(self.maze) - 1 and
                       maze[y][x] != "#" and
                       maze[y][x] != "S"):
                    continue
                movables.append([y, x])
            if len(movables) != 0:
                return movables
            else:
                return None

    def get_val(self, state):
        y, x = state
        if state == self.start_point:
            return 0, False
        else:
            v = float(self.maze[y][x])
            if state == self.goal_point:
                return v, True
            else:
                return v, False


# # Generate a maze

# In[7]:

size = 10
barriar_rate = 0.1

maze_1 = Maze(size, barriar_rate)
maze, start_point, goal_point = maze_1.generate_maze()
maze_field = Field(maze, start_point, goal_point)

maze_field.display()

# # Solving the maze in Deep Q-learning
# ### https://deepmind.com/research/dqn/

# In[55]:


class DQN_Solver:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.e_decay = 0.9999
        self.e_min = 0.01
        self.learning_rate = 0.0001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(2, 2), activation='tanh'))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember_memory(self, state, action, reward, next_state, next_movables, done):
        self.memory.append((state, action, reward, next_state, next_movables, done))

    def choose_action(self, state, movables):
        if self.epsilon >= random.random():
            return random.choice(movables)
        else:
            return self.choose_best_action(state, movables)

    def choose_best_action(self, state, movables):
        best_actions = []
        max_act_value = -100
        for a in movables:
            np_action = np.array([[state, a]])
            act_value = self.model.predict(np_action)
            if act_value > max_act_value:
                best_actions = [a, ]
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def replay_experience(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = []
        Y = []
        for i in range(batch_size):
            state, action, reward, next_state, next_movables, done = minibatch[i]
            input_action = [state, action]
            if done:
                target_f = reward
            else:
                next_rewards = []
                for i in next_movables:
                    np_next_s_a = np.array([[next_state, i]])
                    next_rewards.append(self.model.predict(np_next_s_a))
                np_n_r_max = np.amax(np.array(next_rewards))
                target_f = reward + self.gamma * np_n_r_max
            X.append(input_action)
            Y.append(target_f)
        np_X = np.array(X)
        np_Y = np.array([Y]).T
        self.model.fit(np_X, np_Y, epochs=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay


# In[56]:

state_size = 2
action_size = 2
dql_solver = DQN_Solver(state_size, action_size)

episodes = 20000
times = 1000

for e in range(episodes):
    state = start_point
    score = 0
    for time in range(times):
        movables = maze_field.get_actions(state)
        action = dql_solver.choose_action(state, movables)
        reward, done = maze_field.get_val(action)
        score = score + reward
        next_state = action
        next_movables = maze_field.get_actions(next_state)
        dql_solver.remember_memory(state, action, reward, next_state, next_movables, done)
        if done or time == (times - 1):
            if e % 500 == 0:
                print("episode: {}/{}, score: {}, e: {:.2} \t @ {}"
                      .format(e, episodes, score, dql_solver.epsilon, time))
            break
        state = next_state
    dql_solver.replay_experience(32)


# In[58]:

state = start_point
score = 0
steps = 0
while True:
    steps += 1
    movables = maze_field.get_actions(state)
    action = dql_solver.choose_best_action(state, movables)
    print("current state: {0} -> action: {1} ".format(state, action))
    reward, done = maze_field.get_val(action)
    maze_field.display(state)
    score = score + reward
    state = action
    print("current step: {0} \t score: {1}\n".format(steps, score))
    if done:
        maze_field.display(action)
        print("goal!")
        break
