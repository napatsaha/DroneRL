"""
Test radial raycasting on manual manipulation of drone environment.

General environment performance test.
"""
from envs.environment_v1 import DroneCatch


env = DroneCatch(
    num_preys=0,
    num_predators=1,
    manual_control=True,
    diagnostic=True,
    icon_scale=0.05,
    min_distance=0.0,
    prey_move_speed=2,
    predator_move_speed=8,
    predator_spawn_area=((0.4, 0.4), (0.6, 0.6)),
    prey_spawn_area=((0.2, 0.6), (0.8, 0.7)),
    show_rays=True,
    num_rays=16,
    frame_delay=10,
    obstacle_file="test_codes/obstacle-2.csv",
    num_buffers=2
)

num_eps = 5

for ep in range(num_eps):
    state, info = env.reset()
    # print(state)
    custom = env.render()


    done = False
    while not done:
        action = env.sample_action()
        if env.manual_control:
            action[0] = custom

        state, reward, terminated, truncated, info = env.step(action)
        # print(state)
        # print(reward)
        custom = env.render()


        done = terminated | truncated

        print("--------------")

env.close()