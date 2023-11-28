"""
Test extend line by radius.

Same as test-env-v1-3.py except each line in the obstacle list is extended manually by the agent's radius.
"""
from envs.environment_v1 import DroneCatch


env = DroneCatch(
    num_preys=0,
    num_predators=1,
    manual_control=True,
    diagnostic=True,
    icon_scale=0.05,
    min_distance=0.1,
    prey_move_speed=2,
    predator_move_speed=2,
    show_rays=True,
    num_rays=16,
    frame_delay=10,
    obstacle_file="test_codes/obstacle.csv"
)

env.obstacle_list = [obs.extends(env.active_agents[0].radius) for obs in env.obstacle_list]
env._update_obstacles()

for ep in range(5):
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

env.close()