"""RL: Training an agent to play tic-tac-toe
Use case overview
In this RL scenario, we train an agent to play the game of tic-tac-toe. The agent interacts with the game environment by placing its marks (X or O) on the board and receives rewards for winning (+1), losing (-1), or drawing (0). There are no labeled data or predefined strategies; the agent must learn through trial and error to improve its gameplay.

Data
State: the current configuration of the tic-tac-toe board

Actions: the available positions where the agent can place its mark

Reward: +1 for a win, -1 for a loss, and 0 for a draw

Solution approach
We can use a Q-learning algorithm to train the agent. The agent will play multiple games, and based on the outcomes, it will update its policy to maximize its chances of winning in future games.

Steps involved
Define the environment: the tic-tac-toe board, possible moves, and rules.

Initialize Q-table: store the Q-values for each state-action pair.

Train the agent: the agent plays multiple games, updating its Q-values based on rewards from winning, losing, or drawing.

Evaluate the agent: after training, evaluate the agent’s performance by playing it against a human player or another trained agent."""

import numpy as np

# Initialize Q-table with zeros for all state-action pairs
Q_table = np.zeros((9, 9))  # 9 possible states (board positions) and 9 possible actions

# Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Sample function to select action using epsilon-greedy policy
def epsilon_greedy_action(state, Q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, 9)  # Random action (explore)
    else:
        return np.argmax(Q_table[state])  # Best action (exploit)

# Update Q-values after each game (simplified example)
def update_q_table(state, action, reward, next_state, Q_table):
    Q_table[state, action] = Q_table[state, action] + alpha * (
        reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
    )

# Example simulation of a game where the agent learns
for episode in range(1000):
    state = np.random.randint(0, 9)  # Random initial state
    done = False
    while not done:
        action = epsilon_greedy_action(state, Q_table, epsilon)
        next_state = np.random.randint(0, 9)  # Simulate next state
        reward = 1 if next_state == 'win' else -1 if next_state == 'loss' else 0  # Simulate rewards
        update_q_table(state, action, reward, next_state, Q_table)
        state = next_state
        if reward != 0:
            done = True  # End the game if win/loss



"""Outcome
After playing many games, the agent learns to improve its strategy by adjusting its actions based on the rewards it receives. The agent can eventually play tic-tac-toe competitively, maximizing its chances of winning. RL is used here because the agent learns by interacting with the environment and receiving feedback from game outcomes."""