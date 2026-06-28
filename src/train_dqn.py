import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

from logger import TrainingLogger
from env import FarmersFighterEnv
from models import ConvDQN
from memory import ReplayBuffer

# Use the Nvidia GPU if available. Note: I would recommend an RTX card or better.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 3000
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 50_000
TARGET_UPDATE_FREQ = 100

def epsilon_by_step(global_step):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * global_step / EPS_DECAY)

def optimize_dqn(policy_net, target_net, buffer, optimizer):
    if len(buffer) < BATCH_SIZE:
        return None
    
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    
    # To tensors
    state = torch.from_numpy(state).to(DEVICE).float()
    next_state = torch.from_numpy(next_state).to(DEVICE).float()

    action = torch.from_numpy(action).to(DEVICE).long()
    reward = torch.from_numpy(reward).to(DEVICE).float()
    done = torch.from_numpy(done).to(DEVICE).float()

    # Q(s,a)
    q_values_all = policy_net(state)
    q_values = q_values_all.gather(1, action.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_q_values = target_net(next_state).max(dim=1)[0]
        target = reward + GAMMA * next_q_values * (1.0-done)
        
    loss = nn.MSELoss()(q_values, target)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss

def heuristic_farmer_action(farmer, crops):
    if not crops:
        return np.random.randint(0, 5)  # wander if no crops
    # Find nearest crop
    nearest = min(crops, key=lambda c: abs(c[0]-farmer.x) + abs(c[1]-farmer.y))
    dx = nearest[0] - farmer.x
    dy = nearest[1] - farmer.y
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1  # move down or up
    else:
        return 4 if dy > 0 else 3  # move right or left
    return 0  # stay

# The main training loop!

def train_fighter_dqn():
    env = FarmersFighterEnv()
    fighter_net = ConvDQN(in_channels=6, num_actions=5, grid_size=5).to(DEVICE)
    target_net  = ConvDQN(in_channels=6, num_actions=5, grid_size=5).to(DEVICE)
    target_net.load_state_dict(fighter_net.state_dict())
    target_net.eval()
    
    print("Let's train these bots!")
    
    losses = []
    episode_rewards = []
    optimizer = optim.Adam(fighter_net.parameters(), lr=LR)
    logger = TrainingLogger(log_dir='logs')
    buffer = ReplayBuffer()
    
    global_step = 0
    
    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        t1_f_rewards = 0
        t2_f_rewards = 0
        
        while not done:
            global_step += 1
            epsilon = epsilon_by_step(global_step)
            
            # Fighter observations for both teams
            o_t1 = torch.from_numpy(obs['t1_fighter']).unsqueeze(0).to(DEVICE)
            o_t2 = torch.from_numpy(obs['t2_fighter']).unsqueeze(0).to(DEVICE)
            
            # Actions for fighters
            if np.random.rand() < epsilon:
                a_t1 = np.random.randint(0, 5)
                a_t2 = np.random.randint(0, 5)
            else:
                with torch.no_grad():
                    q_t1 = fighter_net(o_t1)
                    q_t2 = fighter_net(o_t2)
                    a_t1 = int(q_t1.argmax(dim=1).item())
                    a_t2 = int(q_t2.argmax(dim=1).item())
                    
            # Heuristics for the farmer agents
            a_farmer_t1 = heuristic_farmer_action(env.team1_farmer, env.crops)
            a_farmer_t2 = heuristic_farmer_action(env.team2_farmer, env.crops)
            
            actions = {
                't1_farmer': a_farmer_t1,
                't1_fighter': a_t1,
                't2_farmer': a_farmer_t2,
                't2_fighter': a_t2,
            }
            
            next_obs, rewards, done, info = env.step(actions)
            episode_reward += rewards['t1_fighter'] + rewards['t2_fighter']
            t1_f_rewards += rewards['t1_fighter']
            t2_f_rewards += rewards['t2_fighter']
            
            # Storing transitions to the replay buffer
            buffer.push(obs['t1_fighter'], a_t1, rewards['t1_fighter'], next_obs['t1_fighter'], done)
            buffer.push(obs['t2_fighter'], a_t2, rewards['t2_fighter'], next_obs['t2_fighter'], done)
            
            obs = next_obs
            
            if len(buffer) >= BATCH_SIZE:
                loss_val = optimize_dqn(fighter_net, target_net, buffer, optimizer)
                if loss_val is not None:
                    losses.append(loss_val.item())
                    logger.log_loss(global_step, loss_val.item())
                    
            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(fighter_net.state_dict())

        episode_rewards.append(episode_reward)
        logger.log_episode(
            episode=episode,
            total_reward=episode_reward,
            t1_reward=t1_f_rewards,
            t2_reward=t2_f_rewards,
            epsilon=epsilon_by_step(global_step),
            steps=env.step_count,
            buffer_size=len(buffer)
        )
                
        print(f"Episode: {episode}, T1 Fighter: {t1_f_rewards:.2f}, T2 Fighter: {t2_f_rewards:.2f}, reward: {episode_reward:.2f}, step count: {env.step_count} ")
        
        # Periodic network updates
        if episode % 10 == 0:
            target_net.load_state_dict(fighter_net.state_dict())
        
    plot_training(episode_rewards, losses, save_path='training_curves.png')
    logger.close()

    return episode_rewards, losses

def plot_training(episode_rewards, losses, save_path=None):
    episodes = np.arange(len(episode_rewards))

    plt.figure(figsize=(25, 10))

    # Reward curve
    plt.subplot(1, 2, 1)
    plt.plot(episodes, episode_rewards, label='Episode reward')
    if len(episode_rewards) >= 20:
        window = 50
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        plt.plot(range(window - 1, len(episode_rewards)), moving_avg,
                 label=f'{window}-ep avg', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training reward over episodes')
    plt.legend()

    if losses is not None and len(losses) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(losses)
        plt.xlabel('Training step')
        plt.ylabel('Loss')
        plt.title('DQN loss')
    else:
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.text(0.5, 0.5, 'No loss logged', ha='center', va='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
