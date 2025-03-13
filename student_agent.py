# student_agent.py
import numpy as np
import pickle
import random
import os

# ----------------------------
# Global Q-table dictionary
#   Key:   (obs)  (hashable)
#   Value: [Q-value for each action 0..5]
# ----------------------------
Q = {}

QTABLE_FILENAME = "my_qtable.pkl"
EPSILON = 0.05    # Exploration rate
ALPHA   = 0.1     # Learning rate
GAMMA   = 0.99    # Discount factor

# Try to load an existing Q-table if present
if os.path.exists(QTABLE_FILENAME):
    with open(QTABLE_FILENAME, "rb") as f:
        Q = pickle.load(f)
        print("Loaded Q-table from:", QTABLE_FILENAME)
else:
    print("No Q-table found. A new Q-table will be created.")


def get_action(obs):
    """
    Returns an action for the current state (obs) using an epsilon-greedy policy.
    If the obs doesn't exist in Q-table, create it or fallback to random.
    """
    # Make sure obs is in Q-table
    if obs not in Q:
        # Initialize with zeros or small random values
        Q[obs] = [0.0]*6

    # Epsilon-greedy
    if random.random() < EPSILON:
        return random.randint(0, 5)
    else:
        return int(np.argmax(Q[obs]))


def update_qtable(prev_obs, action, reward, next_obs, done):
    """
    Perform one-step Q-learning update:
      Q(S,A) <- Q(S,A) + alpha * [ r + gamma * max_a' Q(S',a') - Q(S,A) ]
    """
    if prev_obs not in Q:
        Q[prev_obs] = [0.0]*6
    if next_obs not in Q:
        Q[next_obs] = [0.0]*6

    old_value = Q[prev_obs][action]
    future_estimate = 0.0 if done else max(Q[next_obs])

    Q[prev_obs][action] = old_value + ALPHA * (reward + GAMMA * future_estimate - old_value)


def save_qtable():
    """Dump the Q-table to a pickle file so it can be reloaded next time."""
    with open(QTABLE_FILENAME, "wb") as f:
        pickle.dump(Q, f)
    print("Q-table saved to", QTABLE_FILENAME)


if __name__ == "__main__":
    """
    Optional Training Loop
    (You can do your training here or in a separate script.)
    """
    from simple_custom_taxi_env import SimpleTaxiEnv

    # Create environment for training
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

    num_episodes = 1000
    max_steps_per_episode = 500

    for ep in range(num_episodes):
        obs, _info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # 1) Choose action
            action = get_action(obs)
            # 2) Step
            next_obs, reward, done, _info = env.step(action)
            # 3) Update Q-table
            update_qtable(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward
            steps += 1

        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward}")

    # Save the Q-table so we can load it next time
    save_qtable()
