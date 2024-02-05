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
    predator_move_speed=2,
    predator_spawn_area=((0.0, 0.7), (0.5, 1.0)),
    prey_spawn_area=((0.4, 0.4), (0.6, 0.6)),
    show_rays=True,
    num_rays=16,
    frame_delay=10,
    obstacle_file="test_codes/obstacle-letterL2.csv",
    num_buffers=2,
    reward_mult=10,
    dist_mult=1,
    include_loc_obs=True,
    include_ray_obs=False,
    include_ray_type_obs=False,
    intermediate_ray_reward="none",
    intermediate_dist_reward=True
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
        print(state)
        print(reward)
        custom = env.render()


        done = terminated | truncated

        print("--------------")

env.close()
