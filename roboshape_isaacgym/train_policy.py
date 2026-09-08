"""
roboshape_isaacgym/train_policy.py
====================================
PPO Training script for 4 Behavioral Validation runs using True Embodied Isaac Gym:
1. Utility - Original
2. Utility - Roboshape
3. Privacy - Original
4. Privacy - Roboshape
"""
from __future__ import annotations
import os
import argparse
import numpy as np

import isaacgym
from scannet_nav_env import TrueEmbodiedNavEnv

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

OBS_DIM_ORIGINAL = 512
OBS_DIM_ROBOSHAPE = 64
ACT_DIM = 2 # vx, vy

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action(self, x):
        mean, val = self.forward(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), val

    def evaluate(self, x, action):
        mean, val = self.forward(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, val, entropy

def train_ppo(env, obs_dim, device="cuda:0", epochs=50, steps_per_epoch=200):
    ac = PPOActorCritic(obs_dim, ACT_DIM).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=3e-4)
    
    # Split envs: 20 bedrooms + 20 non-bedrooms for training, 6 bedrooms + 6 non-bedrooms for testing
    is_bed = env.is_bedroom_tensor.cpu().numpy()
    bed_idx = np.where(is_bed)[0]
    nonbed_idx = np.where(~is_bed)[0]
    
    train_envs = np.concatenate([bed_idx[:20], nonbed_idx[:20]])
    test_envs = np.concatenate([bed_idx[20:], nonbed_idx[20:]])
    
    train_success_rates = []
    test_success_rates = []
    
    # Initial observation
    obs = env.compute_observations()
    
    for epoch in range(epochs):
        
        batch_obs = []
        batch_acts = []
        batch_logp = []
        batch_rews = []
        batch_vals = []
        batch_dones = [] 
        
        train_successes = []
        test_successes = []
        
        # Collect rollout
        for step in range(steps_per_epoch):
            with torch.no_grad():
                if torch.isnan(obs).any():
                    print(f"NAN DETECTED IN OBS at epoch {epoch} step {step}!")
                    break
                action, logp, val = ac.get_action(obs)
                
            next_obs, rew, done, info = env.step(action)
            
            # Record successes (if rew is >= 9.0, it hit the +10 reward)
            is_success = (rew >= 9.0)
            train_successes.extend(is_success[train_envs].cpu().tolist())
            test_successes.extend(is_success[test_envs].cpu().tolist())
            
            batch_obs.append(obs)
            batch_acts.append(action)
            batch_logp.append(logp)
            batch_rews.append(rew)
            batch_vals.append(val.squeeze(-1))
            batch_dones.append(done) # <--- ADD THIS LINE
                
            obs = next_obs
            
        train_success_rate = (sum(train_successes) / len(train_envs))
        test_success_rate = (sum(test_successes) / len(test_envs))
        
        train_success_rates.append(train_success_rate)
        test_success_rates.append(test_success_rate)
        
        # --- NEW: Calculate Discounted Returns ---
        gamma = 0.99
        returns = []
        running_return = torch.zeros(env.num_envs, device=device)
        
        # Loop backward through the batch (list of tensors) to propagate rewards
        for t in reversed(range(len(batch_rews))):
            # ~batch_dones[t] inverts the boolean, multiplies by 0 if done, 1 if not
            running_return = running_return * (~batch_dones[t]).float()
            # Add current reward and discount the future
            running_return = batch_rews[t] + gamma * running_return
            returns.insert(0, running_return.clone())
            
        # PPO Update (simplified) - ONLY ON TRAIN ENVS
        b_obs = torch.cat([obs_step[train_envs] for obs_step in batch_obs], dim=0)
        b_acts = torch.cat([act_step[train_envs] for act_step in batch_acts], dim=0)
        b_logp = torch.cat([logp_step[train_envs] for logp_step in batch_logp], dim=0)
        b_rews = torch.cat([rew_step[train_envs] for rew_step in batch_rews], dim=0)
        b_vals = torch.cat([val_step[train_envs] for val_step in batch_vals], dim=0)
        returns = torch.cat([ret_step[train_envs] for ret_step in returns], dim=0)
        
        # Calculate advantages based on returns, not immediate rewards
        adv = returns - b_vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for _ in range(4): # PPO epochs
            logp, val, ent = ac.evaluate(b_obs, b_acts)
            ratio = torch.exp(logp - b_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (val.squeeze(-1) - returns).pow(2).mean()
            loss = actor_loss + critic_loss - 0.01 * ent.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch:02d} | Train Success: {train_success_rate:.2f} | Test Success: {test_success_rate:.2f}")
        
    return train_success_rates, test_success_rates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    results = {}
    
    print("Initializing True Embodied Isaac Gym Environment...")
    env = TrueEmbodiedNavEnv(num_envs=52, device=args.device)
    env.initialize()
    
    # 1. Utility - Original
    print("\n" + "="*50)
    print(" 1. Training Utility (Wall Avoidance) - Original")
    print("="*50)
    env.set_task(use_roboshape=False, privacy_task=False)
    results["Util_Orig_Train"], results["Util_Orig_Test"] = train_ppo(env, env.num_obs, args.device, epochs=50)
    
    # 2. Utility - Roboshape
    print("\n" + "="*50)
    print(" 2. Training Utility (Wall Seeking) - Roboshape")
    print("="*50)
    env.set_task(use_roboshape=True, privacy_task=False)
    results["Util_Robo_Train"], results["Util_Robo_Test"] = train_ppo(env, env.num_obs, args.device, epochs=50)
    
    # 3. Privacy - Original
    print("\n" + "="*50)
    print(" 3. Training Privacy (Conditional Bedroom Target) - Original")
    print("="*50)
    env.set_task(use_roboshape=False, privacy_task=True)
    results["Priv_Orig_Train"], results["Priv_Orig_Test"] = train_ppo(env, env.num_obs, args.device, epochs=50)
    
    # 4. Privacy - Roboshape
    print("\n" + "="*50)
    print(" 4. Training Privacy (Conditional Bedroom Target) - Roboshape")
    print("="*50)
    env.set_task(use_roboshape=True, privacy_task=True)
    results["Priv_Robo_Train"], results["Priv_Robo_Test"] = train_ppo(env, env.num_obs, args.device, epochs=50)

    # Plot results
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results["Util_Orig_Train"], label="Orig Train", color="#1565C0", linewidth=2)
    plt.plot(results["Util_Orig_Test"], label="Orig Test", color="#90CAF9", linewidth=2, linestyle='--')
    plt.plot(results["Util_Robo_Train"], label="Robo Train", color="#E6A817", linewidth=2)
    plt.plot(results["Util_Robo_Test"], label="Robo Test", color="#FFE082", linewidth=2, linestyle='--')
    plt.title("Phase 1: Utility (Wall Seeking)")
    plt.xlabel("PPO Epoch")
    plt.ylabel("Avg Successes per Env")
    plt.ylim(0, max(max(results["Util_Orig_Train"]), max(results["Util_Robo_Train"])) + 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(results["Priv_Orig_Train"], label="Orig Train", color="#1565C0", linewidth=2)
    plt.plot(results["Priv_Orig_Test"], label="Orig Test", color="#90CAF9", linewidth=2, linestyle='--')
    plt.plot(results["Priv_Robo_Train"], label="Robo Train", color="#E6A817", linewidth=2)
    plt.plot(results["Priv_Robo_Test"], label="Robo Test", color="#FFE082", linewidth=2, linestyle='--')
    plt.title("Phase 2: Privacy (Bedroom Detection)")
    plt.xlabel("PPO Epoch")
    plt.ylabel("Avg Successes per Env")
    plt.ylim(0, max(max(results["Priv_Orig_Train"]), max(results["Priv_Robo_Train"])) + 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("embodied_validation.png", dpi=150)
    print("Saved -> embodied_validation.png")

if __name__ == "__main__":
    main()
