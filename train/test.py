from envs.dual import DualLateral, DualDrone
from envs.environment_v0 import DroneCatch

env = DualLateral(frame_delay=5, min_distance=0.4, verbose=0, trunc_limit=100,
                  prey_move_speed=1, predator_move_speed=10)

num_eps = 10
for i in range(num_eps):
    env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action = []
        for space in env.action_space:
            action.append(space.sample())
        # action = [0,0]
        obs, rew, done, truncated, inf = env.step(action)
        env.render()

env.close()