import copy

from env import Game2048Env
from frame_2048 import draw
import numpy as np

size_board = 4
# env = Game2048Env(size_board)
# env2 = Game2048Env(size_board)
# env2.update_board(env.get_board())


class AI:
    def __init__(self):
        self.__size_board = 4
        self.env = Game2048Env(self.__size_board)

    def make_choice(self):
        pass

    def monty_carlo_choice(self, trials):
        possible_moves = []
        for i in range(4):

            temp_env = Game2048Env(self.__size_board)
            temp_env.step(i)

            if not np.all(np.equal(temp_env.get_board(), self.env.get_board())):
                possible_moves.append(i)
        return max(possible_moves, key=lambda x: self.average_survival(x, trials))

    def average_survival(self, action, trials):
        total = 0
        for i in range(trials):
            total += self.survival_trial(action)
        return total / trials

    def survival_trial(self, action):
        temp_ai = AI()

        temp_ai.env.update_board(self.env.get_board())

        temp_ai.env.step(action)
        return temp_ai.random_play()

    def random_play(self):
        done = 0
        num = 0
        while not done:
            print("done = ", done)
            reward, done = self.env.random_play()
            num += 1
        print("num = ", num)

        return num

    def monty_carlo_play(self, trials):
        done = 0
        while not done:
            action = self.monty_carlo_choice(trials)
            next_state, _, done, _ = self.env.step(action)
            self.env.render()
            if done:
                print("state")
                print(next_state)
                break


if __name__ == '__main__':
    game2 = AI()
    game2.monty_carlo_play(100)
