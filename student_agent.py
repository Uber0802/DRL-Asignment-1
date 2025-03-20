import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

POLICY_FILENAME = "my_policy_net.pth"

POLICY_LR     = 1e-5  # Increased learning rate
GAMMA         = 0.99
HIDDEN_DIM    = 32  # Increased hidden size for better learning
INPUT_DIM     = 8
ACTION_DIM    = 6
GRID_MIN      = 5
GRID_MAX      = 9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def get_dist_and_logits(self, state_tensor):
        logits = self.forward(state_tensor)
        probs  = torch.softmax(logits, dim=1)
        dist   = torch.distributions.Categorical(probs)
        return dist, logits

policy_net = PolicyNet().to(device)

def load_models():
    if os.path.exists(POLICY_FILENAME):
        policy_net.load_state_dict(torch.load(POLICY_FILENAME, map_location=device))
        policy_net.eval()
        print(f"[INFO] Loaded policy net from {POLICY_FILENAME}")
    else:
        print("[INFO] No policy net found; starting untrained.")

def save_models():
    torch.save(policy_net.state_dict(), POLICY_FILENAME)
    print(f"[INFO] Saved policy to {POLICY_FILENAME}")

load_models()

def compress_state(obs):
    """
    Converts the environment's observation into an 8-feature tensor.
    """
    global known_passenger_pos, known_destination_pos, visited_stations
    global passenger_in_taxi  

    (taxi_r, taxi_c,
     s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c,
     obst_n, obst_s, obst_e, obst_w,
     passenger_look, destination_look) = obs

    stations = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]

    if (taxi_r, taxi_c) in stations:
        visited_stations.add((taxi_r, taxi_c))
        if passenger_look and known_passenger_pos == None:
            known_passenger_pos = (taxi_r, taxi_c)
        if destination_look and known_destination_pos == None:
            known_destination_pos = (taxi_r, taxi_c)

    if known_passenger_pos is None or known_destination_pos is None:
        target_r, target_c = next(((r, c) for (r, c) in stations if (r, c) not in visited_stations), stations[0])
    else:
        target_r, target_c = known_passenger_pos if not passenger_in_taxi else known_destination_pos

    rel_target_r = float(target_r - taxi_r)
    rel_target_c = float(target_c - taxi_c)

    can_pickup = 1 if (not passenger_in_taxi) and known_passenger_pos and (taxi_r, taxi_c) == known_passenger_pos else 0
    can_dropoff = 1 if passenger_in_taxi and known_destination_pos and (taxi_r, taxi_c) == known_destination_pos else 0

    feats = [obst_n, obst_s, obst_e, obst_w, rel_target_r, rel_target_c, can_pickup, can_dropoff]

    

    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)


def discount_rewards(rewards, gamma=0.99):
    """
    Compute the discounted sum of rewards.
    """
    discounted = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.append(R)
    discounted.reverse()

    discounted = np.array(discounted)
    discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
    
    return list(discounted)

def get_action_and_logprob(obs):
    state_tensor = compress_state(obs)
    dist, _logits = policy_net.get_dist_and_logits(state_tensor)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


global passenger_in_taxi, known_passenger_pos, visited_stations, known_destination_pos
passenger_in_taxi = False
known_passenger_pos, known_destination_pos  = None, None
visited_stations = set()

def get_action(obs):
    """
    Selects an action based on the current state and updates global variables accordingly.
    This function is used during evaluation.
    """
    global passenger_in_taxi, known_passenger_pos, visited_stations, known_destination_pos

    last_taxi_r, last_taxi_c, *_ = obs  
    if passenger_in_taxi:
        known_passenger_pos = (last_taxi_r, last_taxi_c) 

    state_tensor = compress_state(obs)
    with torch.no_grad():
        dist, _logits = policy_net.get_dist_and_logits(state_tensor)
        action = dist.sample().item()

    if action == 4 and not passenger_in_taxi and (last_taxi_r, last_taxi_c) == known_passenger_pos:
        passenger_in_taxi = True
        known_passenger_pos = None  

    elif action == 5 and passenger_in_taxi:
        passenger_in_taxi = False
        known_passenger_pos = (last_taxi_r, last_taxi_c)  

    

    return action


def train_policy_only(env, policy_net, policy_opt, num_episodes=5000, max_steps=500, gamma=0.99):
    global passenger_in_taxi, known_passenger_pos, known_destination_pos, visited_stations  

    reward_history = []
    success_history = []  
    known_passenger_pos = None
    known_destination_pos = None

    for ep in range(num_episodes):
        obs, _info = env.reset()
        passenger_in_taxi = False
        known_passenger_pos, known_destination_pos = None, None  
        visited_stations = set()

        logprobs = []
        rewards  = []
        done     = False
        total_reward = 0

        while not done and len(rewards) < max_steps:
            last_taxi_r, last_taxi_c, *_ = obs  
            action, log_prob = get_action_and_logprob(obs) 
            # print(action) 
            next_obs, reward, done, info = env.step(action) 
            taxi_r, taxi_c, *_ = next_obs  
            
            
            if action == 4 and passenger_in_taxi == False and (last_taxi_r, last_taxi_c) == known_passenger_pos:
                passenger_in_taxi = True
                known_passenger_pos = (last_taxi_r, last_taxi_c)
            elif action == 5 and passenger_in_taxi == True:
                passenger_in_taxi = False
                known_passenger_pos = (last_taxi_r, last_taxi_c)
            if passenger_in_taxi:
                known_passenger_pos = (taxi_r, taxi_c)

            # print("passenger2 : ", known_passenger_pos)

            logprobs.append(log_prob)
            rewards.append(reward)
            obs = next_obs
            total_reward += reward

        reward_history.append(total_reward)
        success_history.append(info.get("success", False))

        if len(reward_history) > 100:
            reward_history.pop(0)
        if len(success_history) > 100:
            success_history.pop(0)

        returns = discount_rewards(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32, requires_grad=True).to(device)

        policy_opt.zero_grad()
        policy_loss = torch.stack([-lp * Gt for lp, Gt in zip(logprobs, returns)]).sum()
        policy_loss.backward()
        policy_opt.step()

        if (ep + 1) % 100 == 0:
            avg_reward_100 = np.mean(reward_history)
            success_rate = np.mean(success_history) * 100
            print(f"✅ Episode {ep+1}/{num_episodes}, Avg Reward: {avg_reward_100:.2f}, Success Rate: {success_rate:.2f}%")


if __name__ == "__main__":
    from simple_custom_taxi_env import SimpleTaxiEnv
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=POLICY_LR)
    train_policy_only(env, policy_net, policy_optimizer)
    save_models()
