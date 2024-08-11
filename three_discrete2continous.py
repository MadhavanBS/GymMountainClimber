import gymnasium as gym
import numpy as np

env = gym.make('MountainCar-v0', render_mode="human")
env.reset()
done = False

learning_rate = 0.1
discount = 0.95
episodes = 25_000

discrete_observation_space_size = [20] * len(env.observation_space.high)
discrete_observation_space_window_size = (env.observation_space.high-env.observation_space.low)/discrete_observation_space_size

q_table = np.random.uniform(low=-2, high=0, size=discrete_observation_space_size+[env.action_space.n])

def continous_to_discrete(state):
    discrete_state = (state-env.observation_space.low)/discrete_observation_space_size
    return tuple(discrete_state.astype(int))


discrete_state = continous_to_discrete(env.reset()[0])

print(discrete_state)

print(q_table[discrete_state])
print(np.argmax(q_table[discrete_state]))

'''
while not done: 
    action = 2
    new_state, reward, termination, truncation, _ = env.step(action) 
    done = termination or truncation
    # print(f"new_state: {new_state}\nreward: {reward}\ndone: {done}\n")

env.close()
'''