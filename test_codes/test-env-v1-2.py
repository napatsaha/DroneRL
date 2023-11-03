"""

"""
from envs.environment_v1 import DroneCatch


env = DroneCatch(manual_control=True, icon_scale=0.05,
                 prey_move_speed=2,
                 predator_move_speed=2)

for ep in range(5):
    state, info = env.reset()

    done = False
    while not done:
        # action = env.predator.action_space.sample()
        print(state)
        action = env.render()
        state, reward, terminated, truncated, info = env.step(action)

        done = terminated | truncated

env.close()