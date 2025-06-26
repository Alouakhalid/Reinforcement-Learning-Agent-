import gymnasium as gym
import numpy as np 

env = gym.make("Taxi-v3")

alpha = 0.1
gamma  = 0.9
epsilon  = 0.1
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 100000

n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

def epsilon_greedy_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

def update_q_value(state, action, reward, next_state):
    expected_value = np.max(Q[next_state]) 
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * expected_value)

rewards_list = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    print(f"Episode {episode + 1}")

    while not done:
        action = epsilon_greedy_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        update_q_value(state, action, reward, next_state)
        total_reward += reward
        state = next_state

    rewards_list.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


print("Training complete!")
print("Average reward over episodes:", np.mean(rewards_list))
np.save("q_table.npy", Q)



