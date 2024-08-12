import gymnasium as gym
import numpy as np
import ipdb
env = gym.make('MountainCar-v0', render_mode=None)

done = False

learning_rate = 0.1
discount = 0.95
episodes = 25_000


discrete_observation_space_size = [20] * len(env.observation_space.high)
discrete_observation_space_window_size = (env.observation_space.high-env.observation_space.low)/discrete_observation_space_size

epsilon = 0.9 # higher epsilon = higher exploration = high randomness
start_epsilon_decaying = 1 
end_epsilon_decaying = episodes // 2

epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

q_table = np.random.uniform(low=-2, high=0, size=discrete_observation_space_size+[env.action_space.n])

def continous_to_discrete(state):
    discrete_state = (state-env.observation_space.low)/discrete_observation_space_window_size
    rstate = tuple(discrete_state.astype(int))
    # print(f"state: {state}, discrete state: {rstate}")
    return rstate



for episode in range(episodes):

    if episode % 1000 == 0:
        env = gym.make('MountainCar-v0', render_mode="human")
    else:
        env = gym.make('MountainCar-v0', render_mode=None)

    discrete_state = continous_to_discrete(env.reset()[0])
    done = False

    while not done: 
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state, reward, termination, truncation, _ = env.step(action) 
        new_discrete_state = continous_to_discrete(new_state)
        done = termination or truncation
        # print(f"new_state: {new_state}\nreward: {reward}\ndone: {done}\n")
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state+(action,)]
            new_q = (1-learning_rate) * current_q + (learning_rate)* (reward + discount*max_future_q)
            q_table[discrete_state+(action,)] = new_q
            # print(f"old_q: {current_q}, new_q: {new_q}")
        elif new_state[0]>=env.unwrapped.goal_position:
            q_table[discrete_state+(action,)] = 0
            print(f"goal_reached {episode}")

        discrete_state = new_discrete_state
    
    if episode%1000 == 0:
        print(episode)
    
    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value


env.close()
