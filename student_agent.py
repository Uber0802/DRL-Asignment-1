import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

ALPHA = 0.1
GAMMA = 0.99

POLICY_FILENAME = "my_policy_net.pth"


class PolicyNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)  # shape [batch_size, 6]
        return logits

    def get_action(self, state):
        """
        Given a single observation 'state' of shape [1, input_dim],
        return (action, log_prob).
        """
        logits = self.forward(state)           # shape [1, 6]
        probs = torch.softmax(logits, dim=1)  # shape [1, 6]
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()                # shape []
        log_prob = dist.log_prob(action)      # shape []
        return action.item(), log_prob

policy_net = None
optimizer = None

def load_policy_net():
    global policy_net
    if os.path.exists(POLICY_FILENAME):
        policy_net.load_state_dict(torch.load(POLICY_FILENAME))
        policy_net.eval()
        print(f"Loaded policy net from {POLICY_FILENAME}")
    else:
        print("No policy network found. A new one will be created.")

def save_policy_net():
    torch.save(policy_net.state_dict(), POLICY_FILENAME)
    print(f"Policy network saved to {POLICY_FILENAME}")

# ------------------------------------
# Let's keep the clamp/offset approach 
# or do a simpler approach. 
# We'll just flatten the 16-element obs 
# into a float tensor.
# But you can do your compress_state if you want.
# ------------------------------------
def clamp(value, min_val=-3, max_val=3):
    return max(min_val, min(max_val, value))

def compress_state(obs):
    arr = np.array(obs, dtype=np.float32)
    arr = arr.reshape(1, -1)
    return torch.from_numpy(arr)  # shape [1,16]


def get_action(obs, epsilon=0.05):
    state_tensor = compress_state(obs)

    with torch.no_grad():
        logits = policy_net(state_tensor)   
        probs = torch.softmax(logits, dim=1) 
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()              
    return action.item()

def discount_rewards(rewards, gamma=0.99):
    """
    Compute discounted returns for a list of rewards.
    """
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.append(R)
    discounted.reverse()
    return discounted

def train_policy_gradient(env, policy_net, optimizer, num_episodes=1000, max_steps=500, gamma=0.99):
    for ep in range(num_episodes):
        # We'll collect one episode:
        obs, _info = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []

        done = False
        step_count = 0
        total_reward = 0
        while not done and step_count < max_steps:
            state_tensor = compress_state(obs)
            action, log_prob = policy_net.get_action(state_tensor)
            
            # environment step
            passenger_look = obs[-2]
            destination_look = obs[-1]
            next_obs, reward, done, _info = env.step(action, passenger_look, destination_look)
            
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            obs = next_obs
            total_reward += reward
            step_count += 1

        # compute discounted returns
        returns = discount_rewards(rewards, gamma=gamma)
        returns = np.array(returns, dtype=np.float32)
        # optional: normalize returns 
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # policy gradient update
        policy_loss = []
        for (lp, g) in zip(log_probs, returns):
            policy_loss.append(-lp * g)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{num_episodes}, steps={step_count}, total_reward={total_reward:.2f}")

# ------------------------------------
# MAIN 
# We do a PG training loop 
# and then save the policy_net.
# In testing, the environment 
# will just call get_action(), 
# which uses policy_net's forward pass.
# ------------------------------------
if __name__ == "__main__":
    import torch

    from simple_custom_taxi_env import SimpleTaxiEnv

    # 1) Create policy_net
    policy_net = PolicyNet(input_dim=16, hidden_dim=64, output_dim=6)
    # 2) Load existing policy if any
    load_policy_net()

    # 3) Create environment for training
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

    # 4) Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # 5) Train
    num_episodes = 10000
    train_policy_gradient(env, policy_net, optimizer, num_episodes=num_episodes, max_steps=500, gamma=0.99)

    # 6) Save policy
    save_policy_net()
