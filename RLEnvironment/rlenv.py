import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

sizes = 10

episodes = 25000

move_penalty = 1

enemy_penalty = 300

food_reward = 25

epsilon = 0.9

epsilon_decay = 0.9998

show_every = 3000

start_q_table = None

learning_rate = 0.1

discount = 0.95

player_number = 1
food_number = 2
enemy_number = 3

d = {
    1:(255,175,0),
    2:(0,255,0),
    3:(0,0,255),
}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, sizes)
        self.y = np.random.randint(0, sizes)

    def __str__(self):
        return f"({self.x},{self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y, other.y)

    def action(self):
        pass

    def move(self, x=False, y=False)

        