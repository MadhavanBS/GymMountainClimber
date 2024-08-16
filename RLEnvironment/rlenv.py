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

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        
    def move(self, x=False, y=False):
        if not x:
            self.x +=np.random.randint(-1,2)
        else:
            self.x += x

        if not y:
            self.y +=np.randint.randint(-1,2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > sizes-1:
            self.x = sizes-1 

        if self.y < 0:
            self.y = 0
        elif self.y > sizes-1:
            self.y = sizes-1 

if start_q_table is None:
    q_table = {}
    for x1 in range(-sizes+1, sizes):
        for y1 in range(-sizes+1, sizes):
            for x2 in range(-sizes+1, sizes):
                for y2 in range(-sizes+1, sizes):
                    q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5,0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


episode_rewards = []

for episode in range(episodes):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % show_every == 0 :
        print(f"episode: {episode}, epsilon: {epsilon}")
        print(f"mean of last {show_every}: {np.mean(episode_rewards[-show_every:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)
        
        player.action(action)

        if player.x == enemy.x and player.y == enemy_number.y:
            reward = -enemy_penalty
        elif player.x == food.x and player.y == food.y:
            reward = food_reward
        else:
            reward = move_penalty

        new_observation = (player-food, player-enemy)

        max_future_q = np.max(q_table[new_observation]  )

        current_q = q_table[obs][action]

        if reward == food_reward:
            new_q = food_reward
        elif reward == -enemy_penalty:
            new_q = -enemy_penalty
        else:
            new_q = (1-learning_rate)*current_q + (learning_rate)*(reward+discount*max_future_q)

        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.y][food.x] = d[food_number]
            env[player.y][player.x] = d[player_number]
            env[enemy.y][enemy.x] = d[enemy_number]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))
            cv2.imshow("", np.array(img))

            if reward == food_reward or reward == enemy_penalty:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            elif cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward ==  food_reward or reward == enemy_penalty:
            break
        
        episode_rewards.append(episode_reward)
        epsilon*= epsilon_decay

moving_avg = np.convolve(episode_rewards, np.ones((show_every,))/show_every, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward of moving_avg {show_every}")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)







             







    # observation space is 2 co ordiantes, other 2 cordinates x2, y2 (x1,y1) and (x2, y2)


