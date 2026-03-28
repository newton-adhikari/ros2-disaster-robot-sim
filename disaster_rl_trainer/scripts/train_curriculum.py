#!/usr/bin/env python3
import sys, os, time, argparse

import numpy as np
import rclpy
import importlib.util

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


sys.path.insert(0, os.path.dirname(__file__))

# Import from local fixed env file
spec = importlib.util.spec_from_file_location(
    "disaster_nav_env",
    os.path.join(os.path.dirname(__file__), '..', 'disaster_rl_env', 'disaster_nav_env.py')
)
env_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_mod)
DisasterNavEnv = env_mod.DisasterNavEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'tensorboard'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'checkpoints'), exist_ok=True)

# i am using 3 stage curriculum
CURRICULUM = [
    {"name":"stage1_easy",   "timesteps":150_000, "goal_radius":2.0, "description":"Basic goal navigation"},
    {"name":"stage2_medium", "timesteps":250_000, "goal_radius":1.0, "description":"Efficient paths"},
    {"name":"stage3_hard",   "timesteps":200_000, "goal_radius":0.5, "description":"Precision + obstacles"},
]

class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards, self.goals, self.coverages = [], [], []

    def _on_step(self):
        for info, done in zip(self.locals.get('infos',[{}]), self.locals.get('dones',[False])):
            if done:
                ep = info.get('episode', {})
                if ep: self.rewards.append(ep.get('r', 0))
                self.goals.append(int(info.get('goal_reached', False)))
                self.coverages.append(info.get('coverage_cells', 0))
        if self.n_calls % 2000 == 0 and self.rewards:
            n = min(20, len(self.rewards))
            print(f"  [{self.n_calls:7d}] reward={np.mean(self.rewards[-n:]):6.1f} "
                  f"| success={100*np.mean(self.goals[-n:]):4.0f}% "
                  f"| coverage={np.mean(self.coverages[-n:]):4.0f} cells")
        return True

def main():
    p = argparse.ArgumentParser()

    # frequent stop was need, so i have used resume here
    # from last checkpoint
    p.add_argument('--resume', default=None)
    args = p.parse_args()

    rclpy.init()
    ros_node = Node('disaster_trainer')

    print("="*60)
    print("CURRICULUM TRAINING")

    # Wait for Gazebo
    ready = [False]
    def _cb(m): ready[0] = True
    sub = ros_node.create_subscription(LaserScan, '/scan', _cb, 10)
    deadline = time.time() + 30

    while not ready[0] and time.time() < deadline:
        rclpy.spin_once(ros_node, timeout_sec=0.5)
    ros_node.destroy_subscription(sub)

    if not ready[0]:
        print("ERROR: Gazebo not running")
        sys.exit(1)
    print("Gazebo connected\n")

    env_inst = DisasterNavEnv(ros_node)
    env = DummyVecEnv([lambda: Monitor(env_inst)])

    model = PPO(
        "MlpPolicy", env,
        learning_rate  = 3e-4,
        n_steps        = 1024,      # using shorter rollouts
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.02,
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        policy_kwargs  = dict(net_arch=dict(pi=[64,64], vf=[64,64])),
        tensorboard_log= os.path.join(MODELS_DIR, 'tensorboard'),
        verbose        = 1,
        device         = 'auto',
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)

    for i, stage in enumerate(CURRICULUM):
        print(f"\n{'='*60}")
        print(f"STAGE {i+1}: {stage['description']}")
        print(f"  Goal radius: {stage['goal_radius']}m  |  Steps: {stage['timesteps']:,}")
        print(f"{'='*60}")

        env_inst.GOAL_RADIUS = stage['goal_radius']   # type: ignore

        ckpt = CheckpointCallback(
            save_freq=20_000,
            save_path=os.path.join(MODELS_DIR, 'checkpoints'),
            name_prefix=stage['name'],
        )

        model.learn(
            total_timesteps     = stage['timesteps'],
            callback            = [ckpt, EpisodeLogger()],
            progress_bar        = True,
            reset_num_timesteps = (i == 0 and args.resume is None),
        )

        path = os.path.join(MODELS_DIR, stage['name'])
        model.save(path)
        print(f"Stage {i+1} saved: {path}.zip")

    final = os.path.join(MODELS_DIR, 'disaster_ppo_final')
    model.save(final)

    import torch
    pt = final + '.pt'
    torch.save(model.policy, pt)
    print(f"\n Final model: {final}.zip")

    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
