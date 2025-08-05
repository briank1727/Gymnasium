import random
import time


class RowColArmyGame_3x3:

    grid = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    actions = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]

    def __init__(self):
        self.observation_space_n = 2 ** len(self.grid)
        self.max_steps = 200
        self.action_space_n = len(self.actions)

    def action_space_sample(self):
        return random.randint(0, len(self.actions) - 1)

    def reset(self):
        self.state = 0
        self.steps_left = self.max_steps
        while sum(self.grid) == 0:
            for square in range(len(self.grid)):
                if random.uniform(0, 1) < 1 / 2:
                    self.grid[square] = 1
                else:
                    self.grid[square] = 0
                self.state += self.grid[square] * (2**square)

        return self.state

    def step(self, action):
        self.steps_left -= 1
        reward = 0
        for square in range(len(self.grid)):
            if square in self.actions[action]:
                self.state -= self.grid[square] * (2**square)
                self.grid[square] = 0
            else:
                reward -= self.grid[square] * 2

        return self.state, reward, sum(self.grid) == 0, self.steps_left <= 0

    def render(self):
        print(self)
        time.sleep(1)

    def __str__(self):
        output = ""
        output += f"---------\n"
        output += f"|       |\n"
        output += f"| {self.grid[0]} {self.grid[1]} {self.grid[2]} |\n"
        output += f"|       |\n"
        output += f"| {self.grid[3]} {self.grid[4]} {self.grid[5]} |\n"
        output += f"|       |\n"
        output += f"| {self.grid[6]} {self.grid[7]} {self.grid[8]} |\n"
        output += f"|       |\n"
        output += f"---------\n"

        return output
