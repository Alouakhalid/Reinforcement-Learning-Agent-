# 🚕 Taxi-v3 Q-Learning Agent

This project implements a **Reinforcement Learning agent** using **Q-Learning** to solve the classic **Taxi-v3** environment from OpenAI Gymnasium.

---

## 📌 Environment Overview: Taxi-v3

In the **Taxi-v3** environment:

- A taxi agent must **pick up** a passenger from one location
- **Drop them off** at another destination
- Avoid illegal moves and minimize time steps

### Environment Details:

- 🔢 States: 500 discrete states  
- 🎮 Actions: 6 (South, North, East, West, Pickup, Dropoff)  
- 🏁 Goal: Maximize reward by completing the task efficiently

---

## 🧠 RL Algorithm Used

We use **Q-Learning**, a model-free reinforcement learning algorithm that learns an optimal policy using:

- A **Q-Table**: `Q[state, action]` stores expected reward
- **Epsilon-Greedy policy** for exploration vs. exploitation
- **Bellman equation** for Q-value updates

---

## 📦 Installation

```bash
pip install gymnasium numpy
