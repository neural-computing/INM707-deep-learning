# Code from example in https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f

import gym
import numpy as np
import os
import matplotlib.pyplot as plt

# Continuous state space, so no guaratee for convergence of 
# q learning algorithm. Approaches are therefore to either use
# DQNs or to discretise the state space.
env = gym.make("MountainCar-v0")
env.reset()

# Q table in this case is two three-dimensional, as we are 
# working with a 2D state space plus action

# Q-learning function
def q_learning(env, learning, discount, epsilon, min_epsilon, epochs):

    # find size of discretised state space
    num_states = (env.observation_space.high - env.observation_space.low)\
        *np.array([10,100])
    num_states = np.round(num_states, 0).astype(int) + 1

    # initialise Q-table
    # why intialise with random values?
    q_table = np.random.uniform(low=-1, high=1, 
        size=(num_states[0], num_states[1], env.action_space.n))

    # initialise variables to track awards
    reward_list = []
    ave_reward_list = [] # this is to plot

    # Reduction in epsilon by epoch
    reduction = (epsilon - min_epsilon)/epochs

    # Run Q-learning algorithm
    for i in range(epochs):
        # Initialise parameters
        done = False
        # tot_reward is cumulative reward
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretise state
        state_adj = (state - env.observation_space.low)*np.array([10,100])
        state_adj = np.round(state_adj, 0).astype(int)

        while not done:
            # Render environment for last 5 epochs
            if i >= (epochs - 20):
                env.render()
            
            # Determine next action - use epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(q_table[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)
            
            # Get next state and reward
            # "reward" in this case is the observation
            # can ignore "info"
            state2, reward, done, _ = env.step(action)

            # Discretise state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states
            # i.e. cannot roll back down hill
            if done and state2[0] >= 0.5:
                q_table[state2_adj[0], state2_adj[1], action] = reward
            
            # Adjust q-value for current state
            else:
                delta = learning*(reward + 
                    discount*np.max(q_table[state2_adj[0],
                                            state2_adj[1]]) - 
                                    q_table[state_adj[0],
                                            state_adj[1],
                                            action])
                q_table[state_adj[0], state_adj[1], action] += delta
            
            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon -= reduction
        
        reward_list.append(tot_reward)

        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            print(f"Epoch: {(i+1)} Average Reward: {ave_reward}")
            ave_reward_list.append(ave_reward)
            reward_list = []

    env.close()

    return ave_reward_list

# Parameters
learning = 0.2
discount = 0.9
epsilon = 0.8
min_epsilon = 0
epochs = 5000

# Run algorithm
rewards = q_learning(env, learning, discount, epsilon, min_epsilon, epochs)

plt.plot(100*(np.arange(len(rewards))+1), rewards)
plt.xlabel("Epochs")
plt.ylabel("Average Reward")
plt.title("Average Reward by Epoch")
plt.savefig(os.path.join(os.getcwd(),"charts/mountaincar.png"))
plt.close()