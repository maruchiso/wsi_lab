'''
Zaimplementuj algorytm Q-learning. 
Następnie, wykorzystując środowiskoTaxi, zbadaj wpływ hiperparametrów (współczynnik uczenia) 
oraz poznanych strategii eksploracji na działanie algorytmu.
'''
import numpy as np
import gym
import matplotlib.pyplot as plt

def q_learning(env, episodes, gamma, learning_rate, epsilon, epsilon_decay, exploration_strategy):

    #Q and rewards tables inicialization
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        #State is a tuple, the first one contains state value
        state = state[0] 
        total_reward = 0
        done = False

        while not done:
            if exploration_strategy == "epsilon-greedy":
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(Q[state]))
            elif exploration_strategy == "boltzmann":
                epsilon_min = 0.01
                epsilon = max(epsilon, epsilon_min)  
                scaled_Q = Q[state] / epsilon
                scaled_Q = scaled_Q - np.max(scaled_Q)
                probabilities = np.exp(scaled_Q) / np.sum(np.exp(scaled_Q))
                action = int(np.random.choice(env.action_space.n, p=probabilities))

            result = env.step(action)
            next_state, reward, done, truncated, info = result
            #in Taxi-v3 truncated = True, when episode is >= 200
            done = done or truncated
            total_reward += reward

            Q[state, action] += learning_rate * (reward + gamma * Q[next_state, int(np.argmax(Q[next_state]))] - Q[state, action])
            state = next_state

        rewards.append(total_reward)
        epsilon *= epsilon_decay

    return Q, rewards




learning_rates = [0.1, 0.5, 0.9]
strategies = ["epsilon-greedy", "boltzmann"]

results = {}
for strategy in strategies:
    results[strategy] = {}
    for learning_rate in learning_rates:
        Q, rewards = q_learning(env=gym.make('Taxi-v3'), episodes=400, gamma=0.99, learning_rate=learning_rate, epsilon=1.0, epsilon_decay=0.99, exploration_strategy=strategy)
        results[strategy][learning_rate] = rewards

#Drawing plots
for strategy in strategies:
    plt.figure()
    for learning_rate in learning_rates:
        plt.plot(results[strategy][learning_rate], label=f"learning_rate={learning_rate}")
    plt.xlabel("Episode", fontsize=20, fontweight='bold')
    plt.ylabel("Reward", fontsize=20, fontweight='bold')
    plt.title(f"Strategy: {strategy}", fontsize=20, fontweight='bold')
    plt.legend()
    plt.show()
