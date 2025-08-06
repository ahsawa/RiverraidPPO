

import gymnasium as gym
import numpy as np
from typing import SupportsFloat


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob, SEED=10):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState(SEED)
        self.supports_want_render = hasattr(env, "supports_want_render")
        print("STOCHASTIC ACTIVATED")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i == 0:
                rand_ = self.rng.rand()
                if rand_ > self.stickprob:
                    self.curac = ac
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info

class ScoreRewardEnv(gym.Wrapper):
    """
    Reemplaza la recompensa por la diferencia de puntuación
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._prev_value = 0

    @staticmethod
    def _digit(byte_val: int) -> int:
        val = byte_val // 8
        return val if val < 10 else 0

    @classmethod
    def _score_from_info(cls, info: dict) -> int:
        d = cls._digit
        return (
            d(info.get("score0", 0)) +
            d(info.get("score1", 0)) * 10 +
            d(info.get("score2", 0)) * 100 +
            d(info.get("score3", 0)) * 1000 +
            d(info.get("score4", 0)) * 10000 +
            d(info.get("score5", 0)) * 100000
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_value = self._score_from_info(info)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cur_score = self._score_from_info(info)
        reward = cur_score - self._prev_value
        self._prev_value = cur_score
        return obs, reward, terminated, truncated, info

class SoftRewardEnv(gym.RewardWrapper):
    """
    Soft the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        """
        Soft reward

        :param reward:
        :return:
        """
        return float(np.tanh(reward/20))

