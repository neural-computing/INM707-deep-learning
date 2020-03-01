import gym

env = gym.make("Tennis-ram-v0")

# Spaces
# action space: apply force left or right
# observation space: structure of learning environment

for i_episode in range(20):
    observation = env.reset() # trigger "agent-environment" loop

    for t in range(100):
        env.render()
        action = env.action_space.sample()
        # observation: representation of observation from environment
        # reward: amount of reward achieved by previous action
        # done: Boolean, whether episode has terminated
        # info: diagnostic information
        observation, reward, done, info = env.step(action)
        print(observation)
        print(action)
        print("--------------")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
