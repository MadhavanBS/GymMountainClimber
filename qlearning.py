import gymnasium as gym
env = gym.make('MountainCar-v0', render_mode="human")
env.reset()
done = False
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

while not done: 
    action = 2
    new_state, reward, termination, truncation, _ = env.step(action) 
    done = termination or truncation
    # print(f"new_state: {new_state}\nreward: {reward}\ndone: {done}\n")

env.close()

