import random
import time


class RowColArmyGame_2x2:

    grid = [0, 0, 0, 0]

    actions = [[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2]]

    def __init__(self):
        self.observation_space_n = 3 ** len(self.grid)
        self.max_steps = 200
        self.action_space_n = len(self.actions)

    def action_space_sample(self):
        return random.randint(0, len(self.actions) - 1)

    def reset(self):
        self.state = 0
        self.steps_left = self.max_steps
        while sum(self.grid) == 0:
            for square in range(len(self.grid)):
                if random.uniform(0, 1) < 1 / 4:
                    self.grid[square] = 2
                elif random.uniform(0, 1) < 1 / 2:
                    self.grid[square] = 1
                else:
                    self.grid[square] = 0
                self.state += self.grid[square] * (3**square)

        return self.state

    def step(self, action):
        self.steps_left -= 1
        reward = 0
        for square in range(len(self.grid)):
            if square in self.actions[action]:
                self.state -= self.grid[square] * (3**square)
                self.grid[square] = 0
            else:
                reward -= self.grid[square] * 2

        return self.state, reward, sum(self.grid) == 0, self.steps_left <= 0

    def render(self):
        print(self)
        time.sleep(1)

    def __str__(self):
        output = ""
        output += f"-------\n"
        output += f"-------\n"
        output += f"| {self.grid[0]} {self.grid[1]} |\n"
        output += f"|     |\n"
        output += f"| {self.grid[2]} {self.grid[3]} |\n"
        output += f"-------\n"
        output += f"-------\n"

        return output
