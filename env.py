import gym
import numpy as np
import sys
import math
from six import StringIO
from game import Game2048
from gym import spaces
from gym.utils import seeding


class InvalidMove(Exception):
    pass


class Game2048Env(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, size_board, seed=None):
        self.__size_board = size_board
        self.__game = Game2048(size_board)

        self.__zeros = 15
        self.__smooth = 0.9
        self.__var = - 0.9

        # Numbers of possible movements
        self.action_space = spaces.Discrete(4)

        # Numbers of observations
        self.observation_space = spaces.Box(
            0, 2 ** 16, (size_board * size_board,), dtype=np.int
        )

        # Reward range
        self.reward_range = (0., np.inf)

        # Initialise seed
        self.np_random, seed = seeding.np_random(seed)

        # Legends
        self.__actions_legends = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}

        # Old max
        self.__old_max = 0

        # Debug
        self.__last_action = None
        self.__last_scores_move = None

        print("Environment initialised...")

    def __reward_calculation(self, merged):
        # 这里的reward。
        reward = 0
        # 我觉得应该还可以把当前步数来进行计算，步数越多越好
        # 1000个估计跑不出来啥东西～
        # 回头在我电脑跑吧。。
        # 你电脑太慢了。。。9代cpu路过。。
        # 这个部分也要用gpu运算。不然太慢了。，如果不的话，这个部分就需要cpu很强才行
        # 另外这块还有分布式训练，虽然你用不上，但是你知道也是不错的

        cur_board_score = self.__game.get_board().max() * math.log(self.__game.get_board().sum(),2)
                          # self.get_variance(self.__game.get_board()) * self.__var + \

        # 真zz。。。唉。都不更新值? 有什么卵用?
        if cur_board_score > self.__old_max:
            self.__old_max = cur_board_score
            reward += math.log(self.__old_max, 2) * 0.1
        #
        reward += self.get_zeros(self.__game.get_board().flatten()) * self.__zeros

        reward -= self.get_variance(self.__game.get_board()) * self.__var

        reward -= self.get_smooth(self.__game.get_board()) * self.__smooth

        self.__old_max += reward

        reward += merged

        return reward

    def reset(self):
        """Reset the game"""
        self.__game.reset()
        # print("Game reset...")
        valid_movements = np.ones(4)
        return (self.__game.get_board(), valid_movements)

    def step(self, action):
        # print("The enviroment will take a action:", self.__actions_legends[action])
        done = 0
        reward = 0
        try:
            self.__last_action = self.__actions_legends[action]

            self.__game.make_move(action)
            returned_move_scores, returned_merged, valid_movements = (
                self.__game.confirm_move()
            )

            reward = self.__reward_calculation(returned_merged)

            if len(np.nonzero(valid_movements)[0]) == 0:
                done = 1

            self.__last_scores_move = returned_move_scores

            info = dict()
            info["valid_movements"] = valid_movements
            info["total_score"] = self.__game.get_total_score()
            info["last_action"] = self.__actions_legends[action]
            info["scores_move"] = returned_move_scores
            return self.__game.get_board(), reward, done, info

        except InvalidMove as e:
            print("Invalid move")
            done = False
            reward = 0

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        info_render = "Score: {}\n".format(self.__game.get_total_score())
        info_render += "Highest: {}\n".format(self.__game.get_board().max())
        npa = np.array(self.__game.get_board())
        grid = npa.reshape((self.__size_board, self.__size_board))
        info_render += "{}\n".format(grid)
        info_render += "Last action: {}\n".format(self.__last_action)
        info_render += "Last scores move: {}".format(self.__last_scores_move)
        info_render += "\n"
        outfile.write(info_render)
        return outfile

    def get_actions_legends(self):
        return self.__actions_legends

    def get_zeros(self, state):
        """获得面板中0的个数"""
        state = state[state == 0]
        return state.size

    def get_variance(self, state):
        """获得面板的方差"""
        return np.var(state)

    def get_smooth(self, state):
        """获得面板的平滑程度"""
        row = state.T[0].size  # 行数
        col = state[0].size
        sum = 0
        for i, j in enumerate(state):
            for m, n in enumerate(j):
                if 0 < i < row - 1 and m < col - 1:  # 中间部分
                    a = [i - 1, i + 1, i, i]
                    b = [m, m, m - 1, m + 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
                elif i == 0 and m == 0:  # 左上角
                    a = [i + 1, i]
                    b = [m, m + 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
                elif i == 0 and m == col - 1:  # 右上角
                    a = [i + 1, i]
                    b = [m, m - 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
                elif i == row - 1 and m == 0:  # 左下角
                    a = [i - 1, i]
                    b = [m, m + 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
                elif i == row - 1 and m == col - 1:  # 右下角
                    a = [i - 1, i]
                    b = [m, m - 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
                elif i == 0 and 0 < m < col - 1:  # 上
                    a = [i + 1, i, i]
                    b = [m, m + 1, m - 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
                elif i == row - 1 and 0 < m < col - 1:  # 下
                    a = [i - 1, i, i]
                    b = [m, m + 1, m - 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
                elif 0 < i < row - 1 and col == 0:
                    a = [i - 1, i + 1, i]
                    b = [m, m, m + 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
                elif 0 < i < row - 1 and col == 0:
                    a = [i - 1, i + 1, i]
                    b = [m, m, m - 1]
                    array = n - state[a, b]
                    sum += np.log(np.abs(array.sum() + 1))
        return sum