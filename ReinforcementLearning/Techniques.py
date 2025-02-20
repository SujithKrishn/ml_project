import numpy as np
"""In this activity, you will use a simple grid environment (5 × 5) in which the agent starts at a random position and must navigate to a goal state while avoiding pitfalls. The environment includes:

States: each cell on the grid is a unique state.

Actions: the agent can move up, down, left, or right.

Rewards:

+10 for reaching the goal state (position 24).

–10 for falling into a pit (position 12).

–1 for all other movements (to encourage faster goal-reaching).

The objective is to compare how each algorithm—Q-learning and policy gradients—handles this environment and analyze their behavior."""

"""Implement Q-learning
The first part of the activity focuses on Q-learning, a value-based reinforcement learning algorithm."""

# Define the environment (simple example)
n_states = 5  # Number of states
n_actions = 2  # Number of possible actions

# Initialize the Q-table with zeros
Q = np.zeros((n_states, n_actions))

# Set hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Example learning loop
for episode in range(1000):
    state = np.random.randint(0, n_states)  # Random starting state

    done = False
    steps = 0  # Add a step counter to avoid infinite loops
    while not done and steps < 100:  # Limit the number of steps per episode
        # Choose an action using epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q[state, :])  # Exploit known Q-values

        # Simulate action and observe reward and next state
        next_state = np.random.randint(0, n_states)  # Random next state
        reward = np.random.uniform(-1, 1)  # Random reward

        # Update Q-value using Bellman equation
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state  # Move to the next state
        steps += 1  # Increment step counter

        # Example condition to end the episode
        if steps >= 100:  # End episode after 100 steps
            done = True


import numpy as np
import tensorflow as tf

# Define the policy network (simple neural network)
n_states = 4  # Example: 4 input features
n_actions = 2  # Example: 2 possible actions

# Build the policy model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(n_states,)),
    tf.keras.layers.Dense(n_actions, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Function to sample an action based on policy distribution
def get_action(state):
    action_probs = model(state[np.newaxis, :])
    return np.random.choice(n_actions, p=action_probs.numpy()[0])

# Placeholder for rewards and actions
states = []
actions = []
rewards = []

# Example learning loop
for episode in range(1000):
    state = np.random.rand(n_states)  # Example random state

    done = False
    while not done:
        # Sample an action from the policy
        action = get_action(state)
        next_state = np.random.rand(n_states)  # Simulate next state
        reward = np.random.uniform(-1, 1)  # Simulate reward

        # Store trajectory
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        # Break when a stopping condition is met (random here for simplicity)
        if np.random.rand() < 0.1:
            break

    # Compute cumulative rewards
    cumulative_rewards = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        cumulative_rewards[t] = rewards[t] + (0.9 * cumulative_rewards[t+1] if t+1 < len(rewards) else 0)

    # Update policy using the REINFORCE algorithm
    with tf.GradientTape() as tape:
        action_probs = model(np.array(states))
        action_masks = tf.one_hot(actions, n_actions)
        log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs), axis=1)
        loss = -tf.reduce_mean(log_probs * cumulative_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Clear trajectory for next episode
    states, actions, rewards = [], [], []