
from stable_baselines3.dqn import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import numpy as np
import os

from envs.mountain_car import MountainCarWrapper

env = gym.make("MountainCar-v0")
# env = MountainCarWrapper(env)
run_name = "MountainCar_7"

agent = DQN(
    "MlpPolicy", env,
    train_freq=16,
    gradient_steps=8,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=128,
    learning_rate=4e-3,
    policy_kwargs=dict(net_arch=[256, 256])
)

logger = configure(
    os.path.join("logs", "test1", run_name),
    format_strings=["csv", 'stdout']
)
agent.set_logger(logger)

agent.learn(
    total_timesteps=int(1.2e5),
    log_interval=10
)

agent.save(os.path.join("model","test1",run_name))

n_eval=30
ep_rew, ep_len = evaluate_policy(agent, agent.get_env(), n_eval_episodes=n_eval, return_episode_rewards=True)
print(f"evaluation over {n_eval} episodes:")
print(f"Mean reward:\t{np.mean(ep_rew):.3f}")
print(f"Mean length:\t{np.mean(ep_len)}")

## Rendering (Optional)
env = gym.make("MountainCar-v0", render_mode="human")
# env = MountainCarWrapper(env)
agent.set_env(env)
evaluate_policy(agent, agent.get_env(), n_eval_episodes=5, render=True)
env.close()

# num_eps = 20
# vec_env = agent.get_env()
# try:
#     # successes = np.empty((num_eps,), dtype=np.bool_)
#     for i in range(num_eps):
#         done = False
#         obs = vec_env.reset()
#         eps_reward, n_steps = 0, 0
#         while not done:
#             action, _state = agent.predict(obs, deterministic=True)
#             obs, reward, done, info = vec_env.step(action)
#             vec_env.render("human")
#             # print(obs, reward)
#
#             eps_reward += reward
#             n_steps += 1
#             # done = done | truncated
#
#         # successes[i] = info[0]["is_success"]
#
#         # status = "Done" if not truncated else "Truncated"
#         print(f"Episode {i}\t Num Steps: {n_steps}\tReward: {eps_reward}")
#
#     # print(f"\nSuccess Rate:\t{successes.mean():.0%}")
#
#
# finally:
#     vec_env.close()