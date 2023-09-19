from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium import ObservationWrapper, spaces
from gymnasium.core import ObsType, WrapperObsType
import numpy as np
import matplotlib.pyplot as plt

class MountainCarWrapper(ObservationWrapper):
    def __init__(self, env, include_velocity=True):
        assert isinstance(env.unwrapped.unwrapped, MountainCarEnv), "Only a Mountain Car environment can be passed to this wrapper."
        super().__init__(env)

        self.include_velocity = include_velocity

        self.min_y = 0.1
        self.max_y = 1.0

        if self.include_velocity:
            self.low = np.append(env.low, self.min_y).astype(env.low.dtype)
            self.high = np.append(env.high, self.max_y).astype(env.high.dtype)

        else:
            self.low = np.append(self.min_position, self.min_y).astype(env.low.dtype)
            self.high = np.append(self.max_position, self.max_y).astype(env.high.dtype)

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def observation(self, observation: ObsType) -> WrapperObsType:
        x_position = observation[0]
        y_position = self.height(x_position)
        y_position = np.clip(y_position, self.min_y, self.max_y)

        if self.include_velocity:
            return (*observation, y_position)
        else:
            return (x_position, y_position)

if __name__ == "__main__":
    env = MountainCarWrapper(MountainCarEnv())

    # env._height(0)
    #
    # xs = np.linspace(env.min_position, env.max_position, 100)
    # ys = env._height(xs)
    #
    # plt.plot(xs, ys)
    # plt.show()
    obs, _ = env.reset()
    obs_container = []
    n = 1000
    for i in range(n):
        act = env.action_space.sample()
        obs, _, _, _, _ = env.step(act)
        obs = obs[0], obs[2]
        obs_container.append(obs)
    obs_container = np.array(obs_container)

    plt.scatter(obs_container[:,0],
                obs_container[:,1])
    plt.show()