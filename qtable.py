import gymnasium as gym
import time
import numpy as np

env = gym.make('MountainCar-v0', render_mode="human")
env.reset()
done = False
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
'''
[0.6  0.07]
[-1.2  -0.07]
3
split -1.2 to 0.6 to 20 chunks
split -0.07 to 0.07 to 20 chunks
'''

discrete_observation_space_size = [20] * len(env.observation_space.high)
discrete_observation_space_window_size = (env.observation_space.high-env.observation_space.low)/discrete_observation_space_size
print(discrete_observation_space_window_size)

# q_table = np.random.uniform(low=-2, high=0, size=discrete_observation_space_size)
q_table = np.random.uniform(low=-2, high=0, size=discrete_observation_space_size+[env.action_space.n])
print(q_table)
print(q_table.shape)
'''
while not done: 
    action = 2
    new_state, reward, done, _, _ = env.step(action)
    # new_state, *others = env.step(action)
    # print(f"new_state: {new_state}\nreward: {reward}\ndone: {done}\n")
    env.render()

env.close()
'''