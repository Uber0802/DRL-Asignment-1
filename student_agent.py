# student_agent.py
import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------------------------
# Hyperparameters, filenames, etc.
# ------------------------------------------------------------------------------
POLICY_FILENAME = "my_policy_net.pth"
LEARNING_RATE   = 1e-3       # for Adam
ALPHA           = 0.1        # if needed for something else
GAMMA           = 0.99       # discount factor for PG
HIDDEN_DIM      = 64         # hidden size
INPUT_DIM       = 16         # from get_state() in your environment
ACTION_DIM      = 6          # 6 discrete actions

# ------------------------------------------------------------------------------
# Define the Policy Network
# ------------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: [batch_size, input_dim]
        Returns a tensor of shape [batch_size, output_dim] = logits
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action_and_logprob(self, state):
        """
        For training usage:
          Given a single observation 'state' of shape [1, input_dim],
          return (action, log_prob_of_that_action).
        """
        logits = self.forward(state)           # shape [1, 6]
        probs  = torch.softmax(logits, dim=1)  # shape [1, 6]
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()                 # shape []
        log_prob = dist.log_prob(action)       # shape []
        return action.item(), log_prob

# ------------------------------------------------------------------------------
# Global policy_net: created immediately so environment can call get_action()
# ------------------------------------------------------------------------------
policy_net = PolicyNet(INPUT_DIM, HIDDEN_DIM, ACTION_DIM)

def load_policy_net():
    """Load weights from POLICY_FILENAME if it exists, else do nothing special."""
    if os.path.exists(POLICY_FILENAME):
        policy_net.load_state_dict(torch.load(POLICY_FILENAME))
        policy_net.eval()
        print(f"Loaded policy net from {POLICY_FILENAME}")
    else:
        print("No policy network found on disk; starting with an untrained net.")

def save_policy_net():
    """Save current policy_net weights to disk."""
    torch.save(policy_net.state_dict(), POLICY_FILENAME)
    print(f"Policy network saved to {POLICY_FILENAME}")

# Attempt to load the policy net right away (if it exists):
load_policy_net()

# ------------------------------------------------------------------------------
# State Preprocessing
# ------------------------------------------------------------------------------
def compress_state(obs):
    """
    We take the environment's 16-element tuple (taxi_row, taxi_col, etc.)
    and just flatten it into a 1x16 float tensor. 
    (You could do clamp-based offsets or other transformations here if you like.)
    """
    arr = np.array(obs, dtype=np.float32).reshape(1, -1)  # shape [1,16]
    return torch.from_numpy(arr)  # shape [1,16]

# ------------------------------------------------------------------------------
# The environment will call this get_action(obs) in test time
# ------------------------------------------------------------------------------
def get_action(obs, epsilon=0.05):
    """
    Called by the environment to pick an action.
    This version does a forward pass in the policy network and samples 
    from the categorical distribution (i.e., a stochastic policy).
    
    If you prefer a purely greedy approach, replace the sampling with argmax.
    """
    state_tensor = compress_state(obs)

    with torch.no_grad():
        logits = policy_net(state_tensor)             # [1,6]
        probs  = torch.softmax(logits, dim=1)         # [1,6]
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()                        # shape []
    return action.item()

# ------------------------------------------------------------------------------
# Discounted Returns Helper
# ------------------------------------------------------------------------------
def discount_rewards(rewards, gamma=0.99):
    """
    Given a list of rewards for one episode, compute discounted returns G_t.
    """
    discounted = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.append(R)
    discounted.reverse()
    return discounted

# ------------------------------------------------------------------------------
# Training: REINFORCE / Policy Gradient
# ------------------------------------------------------------------------------
def train_policy_gradient(env, policy_net, optimizer, num_episodes=1000, max_steps=500, gamma=0.99):
    for ep in range(num_episodes):
        obs, _info = env.reset()

        # Lists to store per-step data:
        log_probs = []
        rewards   = []
        step_count = 0
        total_reward = 0
        done = False

        while not done and step_count < max_steps:
            state_tensor = compress_state(obs)
            action, log_prob = policy_net.get_action_and_logprob(state_tensor)

            # Step the environment
            passenger_look   = obs[-2]
            destination_look = obs[-1]
            next_obs, reward, done, _info = env.step(action, passenger_look, destination_look)

            log_probs.append(log_prob)
            rewards.append(reward)

            obs = next_obs
            total_reward += reward
            step_count   += 1

        # Compute discounted returns
        returns = discount_rewards(rewards, gamma=gamma)
        returns = np.array(returns, dtype=np.float32)

        # Optionally normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy Gradient Update
        policy_loss = []
        for lp, g in zip(log_probs, returns):
            policy_loss.append(-lp * g)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{num_episodes} | steps={step_count} | total_reward={total_reward:.2f}")

# ------------------------------------------------------------------------------
# If we run this file directly: train the policy net, then save it.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from simple_custom_taxi_env import SimpleTaxiEnv

    # Create environment (5x5 as example)
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

    # Set up optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    # Train
    num_episodes = 1000
    train_policy_gradient(env, policy_net, optimizer, num_episodes=num_episodes, max_steps=500, gamma=GAMMA)

    # Save final policy
    save_policy_net()
