import gymnasium as gym
import numpy as np

env = gym.make('MountainCar-v0', render_mode="human")
env.reset()
done = False


discrete_observation_space_size = [20] * len(env.observation_space.high)
discrete_observation_space_window_size = (env.observation_space.high-env.observation_space.low)/discrete_observation_space_size

q_table = np.random.uniform(low=-2, high=0, size=discrete_observation_space_size+[env.action_space.n])

'''
while not done: 
    action = 2
    new_state, reward, done, _, _ = env.step(action)
    # new_state, *others = env.step(action)
    # print(f"new_state: {new_state}\nreward: {reward}\ndone: {done}\n")

env.close()
'''