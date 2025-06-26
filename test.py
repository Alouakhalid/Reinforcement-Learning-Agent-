import gymnasium as gym
import numpy as np 

env = gym.make("Taxi-v3")
test_episodes = 3
test_rewards = []

env = gym.make("Taxi-v3", render_mode="human")  
Q = np.load("q_table.npy")
for episode in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q[state])  
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

    test_rewards.append(total_reward)

env.close()
print("✅ Evaluation done.")
print("🎯 Average reward over 100 test episodes:", np.mean(test_rewards))
