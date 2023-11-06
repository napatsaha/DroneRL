"""
Test radial raycasting on manual manipulation of drone environment.
"""
from envs.environment_v1 import DroneCatch


env = DroneCatch(manual_control=True, icon_scale=0.05,
                 prey_move_speed=2,
                 predator_move_speed=2,
                 show_rays=True,
                 num_rays=6)

for ep in range(5):
    state, info = env.reset()
    action = env.render()
    print(state)

    done = False
    while not done:
        # action = env.predator.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)
        action = env.render()
        print(state)

        done = terminated | truncated

env.close()