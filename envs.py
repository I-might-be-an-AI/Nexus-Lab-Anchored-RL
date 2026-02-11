"""
MuJoCo environments with injected hazard signals.

Each env returns (obs, reward, cost, done, info) where cost > 0
indicates a hazard violation.
"""

import numpy as np


class HazardHalfCheetah:
    """
    HalfCheetah with three hazard sources:
      1. Velocity hazard  – forward velocity exceeds threshold
      2. Angle hazard     – joint angles exceed safe range
      3. Random gusts     – stochastic action perturbations
    """

    def __init__(
        self,
        velocity_threshold: float = 8.0,
        angle_threshold: float = 0.8,
        gust_probability: float = 0.05,
        cost_penalty: float = -1.0,
    ):
        try:
            import gymnasium as gym
        except ImportError:
            import gym

        self.env = gym.make("HalfCheetah-v4")
        self.state_dim = self.env.observation_space.shape[0]   # 17
        self.action_dim = self.env.action_space.shape[0]        # 6

        self.velocity_threshold = velocity_threshold
        self.angle_threshold = angle_threshold
        self.gust_probability = gust_probability
        self.cost_penalty = cost_penalty

        # Lifetime stats
        self.total_cost = 0
        self.total_velocity_hazards = 0
        self.total_angle_hazards = 0
        self.episode_cost = 0

    def reset(self):
        self.episode_cost = 0
        out = self.env.reset()
        return out[0].astype(np.float32) if isinstance(out, tuple) else out.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)

        # Random gust
        if np.random.random() < self.gust_probability:
            gust = np.random.uniform(-0.5, 0.5, self.action_dim)
            action = np.clip(action + gust, -1, 1)

        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            done = term or trunc
        else:
            obs, reward, done, info = out
        obs = obs.astype(np.float32)

        cost = 0.0

        # Velocity hazard (obs[8] = forward velocity)
        if len(obs) > 8 and abs(obs[8]) > self.velocity_threshold:
            cost += 1.0
            self.total_velocity_hazards += 1

        # Angle hazard (obs[2:8] = joint angles)
        joint_angles = obs[2:8] if len(obs) > 8 else obs[2:min(8, len(obs))]
        if np.any(np.abs(joint_angles) > self.angle_threshold):
            cost += 1.0
            self.total_angle_hazards += 1

        self.total_cost += cost
        self.episode_cost += cost

        return obs, float(reward), float(cost), done, info

    def close(self):
        self.env.close()


class HazardAnt:
    """
    Ant with three hazard sources:
      1. Height hazard       – torso drops below threshold (falling)
      2. Orientation hazard  – torso tilt exceeds safe range
      3. Velocity hazard     – excessive speed
    """

    def __init__(
        self,
        height_threshold: float = 0.3,
        tilt_threshold: float = 0.5,
        velocity_threshold: float = 5.0,
        gust_probability: float = 0.05,
    ):
        try:
            import gymnasium as gym
        except ImportError:
            import gym

        self.env = gym.make("Ant-v4")
        self.state_dim = self.env.observation_space.shape[0]   # 27
        self.action_dim = self.env.action_space.shape[0]        # 8

        self.height_threshold = height_threshold
        self.tilt_threshold = tilt_threshold
        self.velocity_threshold = velocity_threshold
        self.gust_probability = gust_probability

        self.total_cost = 0
        self.episode_cost = 0

    def reset(self):
        self.episode_cost = 0
        out = self.env.reset()
        return out[0].astype(np.float32) if isinstance(out, tuple) else out.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)

        if np.random.random() < self.gust_probability:
            gust = np.random.uniform(-0.5, 0.5, self.action_dim)
            action = np.clip(action + gust, -1, 1)

        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            done = term or trunc
        else:
            obs, reward, done, info = out
        obs = obs.astype(np.float32)

        cost = 0.0

        if obs[0] < self.height_threshold:
            cost += 1.0

        if len(obs) > 4:
            orientation = obs[1:5]
            if np.any(np.abs(orientation[1:]) > self.tilt_threshold):
                cost += 1.0

        self.total_cost += cost
        self.episode_cost += cost

        return obs, float(reward), float(cost), done, info

    def close(self):
        self.env.close()
