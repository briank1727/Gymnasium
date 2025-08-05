import env_preset
import numpy as np


class env:
    playerX = 0
    playerY = 0
    playerDir = 0
    steps = 0
    apple_positions = []

    def __init__(self, ep: env_preset):
        self.ep = ep

    def reset(self):
        player_spawn_angle = np.random.random() * np.pi * 2
        radius = np.random.random() * self.ep.player_pos_randomness
