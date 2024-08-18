import numpy as np
import tensorflow.keras.backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

replay_memory_size = 50_000
min_replay_memory_size = 1_000
model_name = "2x256"
mini_batch_size = 64
discount = 0.99
update_target_every = 5
min_reward = -200
memory_fraction = 0.2 

episodes = 20_000

epsilon = 1
epsilon_decay = 0.999975
min_epsilon = 0.001

aggregate_stats_every = 50
show_preview = False


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

class BlobEnv:
    sizes = 10
    return_images = True
    move_penalty = 1
    enemy_penalty = 300
    food_reward = 25
    observation_space_values = (sizes, sizes, 3)  # 4
    action_space_size = 9
    player_n = 1  # player key in dict
    food_n = 2  # food key in dict
    enemy_n = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.sizes)
        self.food = Blob(self.sizes)
        while self.food == self.player:
            self.food = Blob(self.sizes)
        self.enemy = Blob(self.sizes)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.sizes)

        self.episode_step = 0

        if self.return_images:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.return_images:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.enemy_penalty
        elif self.player == self.food:
            reward = self.food_reward
        else:
            reward = -self.move_penalty

        done = False
        if reward == self.food_reward or reward == -self.enemy_penalty or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.sizes, self.sizes, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.food_n]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.enemy_n]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.player_n]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img



env = BlobEnv()
ep_rewards = [-200]

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, model_name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class DQNAgent:

    def __init__(self):
        # gets trained every step
        self.model = self.create_model()

        # we predict against this model every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen = replay_memory_size)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{model_name}-{int(time.time())}") 

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=env.observation_space_values))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.action_space_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, terminal_state, step):
        return seld.model_predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < min_replay_memory_size:
            return

        mini_batch = random.sample(self.replay_memory, mini_batch_size)

        current_states = np.array([transition[0] for transition in mini_batch])/ 255

        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in mini_batch])/255

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]

            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(y), batch_size=mini_batch_size,
        verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_update_counter+=1

        if self.target_update_counter > update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



if not os.path.isdir('models'):
    os.makedirs('models')

agent = DQNAgent()

for episode in tqdm(range(1, episodes+1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode 

    episode_reward = 0
    
    step = 1

    current_state = env.reset()

    done =  False

    while not done:
        if np.random.random() > epsilon: 
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space_size)

        new_state, reward, done = env.step(action)

        episode_reward += reward

        if show_preview and not episode % aggregate_stats_every:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state

        step+=1


    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % aggregate_stats_every or episode == 1:
        average_reward = sum(ep_rewards[-aggregate_stats_every:])/len(ep_rewards[-aggregate_stats_every:])
        mini_reward = min(ep_rewards[-aggregate_stats_every:])
        max_reward = max(ep_rewards[-aggregate_stats_every:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if mini_reward >= min_reward:
            agent.model.save(f'models/{model_name}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)
