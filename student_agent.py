# student_agent.py

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

POLICY_FILENAME = "my_policy_net.pth"
VALUENET_FILENAME = "my_value_net.pth"

POLICY_LR     = 1e-4  
VALUE_LR      = 1e-4   
GAMMA         = 0.99   
HIDDEN_DIM    = 64     
INPUT_DIM     = 10     
ACTION_DIM    = 6      
GRID_MIN      = 5      
GRID_MAX      = 9      

def clamp(value, min_val=-5, max_val=5):
    return max(min_val, min(max_val, value))

class PolicyNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: [batch_size, input_dim=14]
        returns logits: [batch_size, 6]
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action_logprob(self, state):
        """
        For training:
        input: state shape [1,14]
        returns: (action, log_prob(action))
        """
        logits = self.forward(state)           
        probs  = torch.softmax(logits, dim=1)  
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()                 
        log_prob = dist.log_prob(action)       
        return action.item(), log_prob

class ValueNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [batch_size, input_dim=8]
        returns value: [batch_size, 1]
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.v(x)
        return value

policy_net = PolicyNet(INPUT_DIM, HIDDEN_DIM, ACTION_DIM)
value_net  = ValueNet(INPUT_DIM, HIDDEN_DIM)

def load_models():
    if os.path.exists(POLICY_FILENAME):
        policy_net.load_state_dict(torch.load(POLICY_FILENAME))
        policy_net.eval()
        print(f"[INFO] Loaded policy net from {POLICY_FILENAME}")
    else:
        print("[INFO] No policy net found; starting untrained.")

    if os.path.exists(VALUENET_FILENAME):
        value_net.load_state_dict(torch.load(VALUENET_FILENAME))
        value_net.eval()
        print(f"[INFO] Loaded value net from {VALUENET_FILENAME}")
    else:
        print("[INFO] No value net found; starting untrained.")

def save_models():
    torch.save(policy_net.state_dict(), POLICY_FILENAME)
    torch.save(value_net.state_dict(), VALUENET_FILENAME)
    print(f"[INFO] Saved policy to {POLICY_FILENAME}, value to {VALUENET_FILENAME}")

load_models()


known_passenger_pos = None
known_destination_pos = None
visited_stations = set() 

# 記錄已知的 passenger 和 destination 位置
known_passenger_pos = None
known_destination_pos = None
visited_stations = set()  # 記錄已探索的車站

def compress_state(obs):
    """
    Features:
    1️⃣ Four obstacle indicators (4)
    2️⃣ Target direction indicators (4)
    3️⃣ Can Pick Up (1)
    4️⃣ Can Drop Off (1)
    """
    global known_passenger_pos, known_destination_pos, visited_stations

    (taxi_r, taxi_c,
     s0_r, s0_c,
     s1_r, s1_c,
     s2_r, s2_c,
     s3_r, s3_c,
     obst_n, obst_s, obst_e, obst_w,
     passenger_look, destination_look) = obs

    stations = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]

    # 1️⃣ Update Knowledge (Track Passenger & Destination)
    if (taxi_r, taxi_c) in stations:
        visited_stations.add((taxi_r, taxi_c))
        if passenger_look:
            known_passenger_pos = (taxi_r, taxi_c)
        if destination_look:
            known_destination_pos = (taxi_r, taxi_c)

    # 2️⃣ Determine Target
    if len(visited_stations) < 4:
        target_r, target_c = min(stations, key=lambda s: abs(s[0] - taxi_r) + abs(s[1] - taxi_c))
    elif known_passenger_pos is None:
        target_r, target_c = min(stations, key=lambda s: abs(s[0] - taxi_r) + abs(s[1] - taxi_c))
    elif (taxi_r, taxi_c) == known_passenger_pos:
        target_r, target_c = known_destination_pos if known_destination_pos else stations[0]
    else:
        target_r, target_c = known_passenger_pos

    # 3️⃣ Compute Directional Indicators
    target_n = 1 if target_r < taxi_r else 0
    target_s = 1 if target_r > taxi_r else 0
    target_e = 1 if target_c > taxi_c else 0
    target_w = 1 if target_c < taxi_c else 0

    # 4️⃣ Pick-Up & Drop-Off Indicators
    can_pickup = 1 if (taxi_r, taxi_c) == known_passenger_pos else 0
    can_dropoff = 1 if (taxi_r, taxi_c) == known_destination_pos else 0

    # 5️⃣ Final State Representation (10 Features)
    feats = [
        obst_n, obst_s, obst_e, obst_w,
        target_n, target_s, target_e, target_w,
        can_pickup, can_dropoff
    ]
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # Shape [1,10]




# ------------------------------------------------------------------------------
# The environment calls get_action(obs) to pick an action at test time
# ------------------------------------------------------------------------------
def get_action(obs, epsilon=0.2):  # Increase epsilon for more exploration
    """
    1️⃣ First explore all four stations.
    2️⃣ Move toward passenger, then toward destination.
    3️⃣ Use policy net to make smart decisions.
    """
    global known_passenger_pos, visited_stations

    state_tensor = compress_state(obs)

    with torch.no_grad():
        logits = policy_net(state_tensor)
        probs  = torch.softmax(logits, dim=1)
        dist   = torch.distributions.Categorical(probs)

        # Epsilon-Greedy Exploration
        if random.random() < epsilon:
            action = random.randint(0, 5)  
        else:
            action = dist.sample().item()

    # 🚖 Ensure Early Exploration
    if len(visited_stations) < 4:
        action = random.choice([0, 1, 2, 3])

    print(f"🔍 Taxi State: {state_tensor.numpy().flatten()}")
    print(f"🚖 Chosen Action: {action}")

    return int(action)





# ------------------------------------------------------------------------------
# discount_rewards for a single trajectory
# ------------------------------------------------------------------------------
def discount_rewards(rewards, gamma=0.99):
    discounted = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma*R
        discounted.append(R)
    discounted.reverse()
    return discounted

# ------------------------------------------------------------------------------
# Training with advantage = G_t - V(s_t)
# We do a single step of gradient for policy and value net each episode
# ------------------------------------------------------------------------------
def train_with_advantage(env, policy_net, value_net, policy_opt, value_opt, num_episodes=5000, max_steps=500, gamma=0.99, value_loss_coef=0.5):
    """
    1️⃣ Uses Advantage = G_t - V(s_t)
    2️⃣ Normalizes rewards for stability
    3️⃣ Tracks success rate
    4️⃣ Saves model when average reward > 0
    """
    successful_episodes = 0  # Track successful drop-offs
    reward_history = []  # Store last 100 rewards for averaging

    for ep in range(num_episodes):
        obs, _info = env.reset()

        states = []
        logprobs = []
        rewards = []
        step = 0
        done = False
        total_reward = 0
        success = False  

        while not done and step < max_steps:
            st = compress_state(obs)

            action, log_prob = policy_net.get_action_logprob(st)

            # Step in the environment
            passenger_look = obs[-2]
            destination_look = obs[-1]
            next_obs, reward, done, _info = env.step(action, passenger_look, destination_look)

            states.append(st)
            logprobs.append(log_prob)
            rewards.append(reward)

            obs = next_obs
            total_reward += reward
            step += 1

            # ✅ Check for successful episode (passenger dropped at destination)
            if done and reward >= 50:
                success = True

        # ✅ Update success count
        if success:
            successful_episodes += 1

        # ✅ Store reward history for averaging
        reward_history.append(total_reward)
        if len(reward_history) > 100:
            reward_history.pop(0)  # Keep only last 100 rewards

        # 🚀 Compute Average Reward
        avg_reward = np.mean(reward_history) if len(reward_history) > 0 else -100

        # 🚀 Normalize & Clip Rewards
        returns = discount_rewards(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)

        policy_opt.zero_grad()
        value_opt.zero_grad()

        policy_loss = []
        value_loss = []

        for i, (lp, Gt) in enumerate(zip(logprobs, returns)):
            v_s = value_net(states[i])
            advantage = Gt - v_s.item()

            policy_loss.append(-lp * advantage)
            value_loss.append(0.5 * advantage**2)

        # ✅ Convert to Tensors that Track Gradients
        policy_loss = torch.stack(policy_loss).sum()
        value_loss = torch.stack(value_loss).sum()  

        total_loss = policy_loss + value_loss_coef * value_loss
        total_loss.backward()

        policy_opt.step()
        value_opt.step()

        # ✅ Save Model if Avg Reward > 0
        if avg_reward > 0:
            save_models()
            print(f"📌 Model Saved! ✅ Avg Reward: {avg_reward:.2f}")

        # ✅ Print success rate and avg reward every 100 episodes
        if (ep + 1) % 100 == 0:
            success_rate = (successful_episodes / (ep + 1)) * 100
            print(f"✅ Episode {ep+1}/{num_episodes}, Steps: {step}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}%")



# ------------------------------------------------------------------------------
# If run directly: do advantage training, save models
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Create the environment
    base_env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

    # 1) Grab the original reset method:
    original_reset = base_env.reset

    # 2) Define a new function that calls original_reset
    def random_reset():
        size = random.randint(GRID_MIN, GRID_MAX)
        base_env.grid_size = size
        return original_reset()

    # 3) Monkey-patch env.reset:
    base_env.reset = random_reset

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=POLICY_LR)
    value_optimizer  = optim.Adam(value_net.parameters(),  lr=VALUE_LR)

    num_episodes = 5000
    train_with_advantage(base_env,
                         policy_net, value_net,
                         policy_optimizer, value_optimizer,
                         num_episodes=num_episodes,
                         max_steps=1000,
                         gamma=GAMMA,
                         value_loss_coef=0.5)

    save_models()