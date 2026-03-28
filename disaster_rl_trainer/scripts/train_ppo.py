#!/usr/bin/env python3


import sys, os, time, argparse
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback
)
from disaster_rl_env.disaster_nav_env import DisasterNavEnv
import torch
import torch.nn.functional as F
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(os.path.join(MODELS_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'tensorboard'), exist_ok=True)


class ProgressCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards  = []
        self.ep_success  = []
        self.ep_coverage = []
        self.last_print  = 0

    def _on_step(self):
        for info, done in zip(self.locals.get('infos',[{}]),
                              self.locals.get('dones',[False])):
            if done:
                ep = info.get('episode', {})
                if ep:
                    self.ep_rewards.append(ep.get('r', 0))
                self.ep_success.append(int(info.get('goal_reached', False)))
                self.ep_coverage.append(info.get('coverage_cells', 0))

        if self.n_calls - self.last_print >= 2000 and self.ep_rewards:
            n = min(10, len(self.ep_rewards))
            r    = np.mean(self.ep_rewards[-n:])
            succ = np.mean(self.ep_success[-n:]) * 100
            cov  = np.mean(self.ep_coverage[-n:])
            fps  = self.n_calls / max(1, time.time() - self._start)
            bar  = "▓" * int(succ / 5)
            print(f"[{self.n_calls:7d}] reward={r:6.0f} "
                  f"success={succ:4.0f}% {bar:<20s} "
                  f"cov={cov:4.0f}cells fps={fps:.0f}")
            self.last_print = self.n_calls
        return True

    def _on_training_start(self):
        self._start = time.time()


def collect_reactive_demos(env, n_steps=5000):
    
    print(f"Collecting {n_steps} reactive policy demonstrations...")

    obs_list, act_list = [], []
    obs, _ = env.reset()

    for _ in range(n_steps):
        # Replicate reactive policy logic 
        sectors = obs[:12] * 12.0   # denormalise
        bearing = obs[13] * math.pi  # goal bearing

        front = min(sectors[0], sectors[11], sectors[1])
        fl    = min(sectors[10], sectors[11])
        fr    = min(sectors[1],  sectors[2])

        STOP = 0.35
        if front < STOP:
            turn = 1.0 if fl >= fr else -1.0
            action = np.array([0.0, turn], dtype=np.float32)
        else:
            speed = min(1.0, front / 1.5)
            ang   = np.clip(bearing * 0.6, -1.0, 1.0)
            action = np.array([speed, ang], dtype=np.float32)

        obs_list.append(obs.copy())
        act_list.append(action)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    print(f" Collected {len(obs_list)} demo transitions")
    return np.array(obs_list), np.array(act_list)


def pretrain_bc(model, obs_arr, act_arr, n_epochs=5):
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(obs_arr),
        torch.FloatTensor(act_arr)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"Pretraining via behavioural cloning ({n_epochs} epochs)...")
    for epoch in range(n_epochs):
        losses = []
        for obs_b, act_b in loader:
            obs_b = obs_b.to(model.device)
            act_b = act_b.to(model.device)

            dist = policy.get_distribution(obs_b)
            log_prob = dist.log_prob(act_b)
            loss = -log_prob.mean()   # maximise likelihood of reactive actions

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

        print(f" BC Epoch {epoch+1}/{n_epochs}: loss={np.mean(losses):.4f}")

    print(" Behavioural cloning done")


def main(args):
    rclpy.init()
    ros_node = Node('ppo_trainer')

    # Wait for simulation
    ready = [False]
    def _cb(msg): ready[0] = True
    sub = ros_node.create_subscription(LaserScan, '/scan', _cb, 10)

    print("Waiting for Gazebo...")

    timeout = time.time() + 30
    while not ready[0] and time.time() < timeout:
        rclpy.spin_once(ros_node, timeout_sec=0.5)
    ros_node.destroy_subscription(sub)
    if not ready[0]:
        print("ERROR: Gazebo not running")
        sys.exit(1)
    print(" Gazebo connected\n")

    # Build env
    raw_env = DisasterNavEnv(ros_node)
    env = DummyVecEnv([lambda: Monitor(raw_env,
                       filename=os.path.join(MODELS_DIR, 'monitor'))])

    model = PPO(
        "MlpPolicy", env,
        learning_rate  = 1e-4,          # lower → more stable value loss
        n_steps        = 512,           # smaller 
        batch_size     = 32,
        n_epochs       = 5,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        clip_range_vf  = 0.2,           
        ent_coef       = 0.05,          # more exploration
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        policy_kwargs  = dict(
            net_arch          = dict(pi=[64, 64], vf=[64, 64]),
            activation_fn     = __import__('torch').nn.ReLU,
            log_std_init      = -1.0,  
        ),
        tensorboard_log= os.path.join(MODELS_DIR, 'tensorboard'),
        verbose        = 0,
        device         = 'auto',
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, device='auto')
    else:
        demo_obs, demo_act = collect_reactive_demos(raw_env, n_steps=2000)
        pretrain_bc(model, demo_obs, demo_act, n_epochs=10)
        print()

    curriculum = [
        {"steps":  80_000, "goal_r": 2.0, "label": "Stage 1 — Easy (2m goal)"},
        {"steps": 120_000, "goal_r": 1.0, "label": "Stage 2 — Medium (1m goal)"},
        {"steps": 100_000, "goal_r": 0.5, "label": "Stage 3 — Hard (0.5m goal)"},
    ]

    for i, stage in enumerate(curriculum):
        print(f"\n{'='*55}")
        print(f"{stage['label']}")
        print(f"  Steps: {stage['steps']:,}   Goal radius: {stage['goal_r']} m")
        print(f"{'='*55}")

        raw_env.GOAL_RADIUS = stage['goal_r']

        ckpt = CheckpointCallback(
            save_freq  = 10_000,
            save_path  = os.path.join(MODELS_DIR, 'checkpoints'),
            name_prefix= f'disaster_ppo_s{i+1}',
        )
        progress = ProgressCallback()

        model.learn(
            total_timesteps     = stage['steps'],
            callback            = [ckpt, progress],
            progress_bar        = False,  # off → cleaner output
            reset_num_timesteps = (i == 0 and not args.resume),
        )

        stage_path = os.path.join(MODELS_DIR, f'stage{i+1}')
        model.save(stage_path)
        print(f"\n✓ Stage {i+1} saved: {stage_path}.zip")

    final_zip = os.path.join(MODELS_DIR, 'disaster_ppo_final')
    model.save(final_zip)

    pt_path = final_zip + '.pt'
    torch.save(model.policy, pt_path)

    print(f"\n{'='*55}")
    print(f" Training complete!")
    print(f"  Model:  {final_zip}.zip")
    print(f"  Policy: {pt_path}")

    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--resume', default=None, help='Path to checkpoint .zip')
    p.add_argument('--timesteps', type=int, default=None, help='Override total timesteps')
    
    main(p.parse_args())
