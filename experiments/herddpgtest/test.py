from curriculum.envs.maze.maze_ant.ant_maze_env import AntMazeEnv
# from curriculum.envs.maze.maze_ant.ant_target_env import AntEnv
from curriculum.envs.maze.point_maze_env import PointMazeEnv
from curriculum.envs.base import UniformStateGenerator
from rllab.envs.normalized_env import normalize
from curriculum.envs.goal_env import GoalExplorationEnv
from rllab.misc.instrument import VariantGenerator
import numpy as np

vg = VariantGenerator()
vg.add('maze_id', [0])
vg.add('goal_size', [2])
vg.add('goal_range', [5])  # this will be used also as bound of the state_space
vg.add('goal_center', [(0, 0)])
vg.add('terminal_eps', [0.5])
vg.add('distance_metric', ['L2'])
vg.add('extend_dist_rew', [False])
vg.add('only_feasible', [True])
# env = AntMazeEnv()

for v in vg.variants():
    inner_env = normalize(AntMazeEnv(maze_id=v['maze_id']))
    # inner_env = normalize(AntEnv())
    # inner_env = normalize(PointMazeEnv(maze_id=v['maze_id']))
    uniform_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=v['goal_range'],
                                                center=v['goal_center'])
    env = GoalExplorationEnv(
    env=inner_env, goal_generator=uniform_goal_generator,
    obs2goal_transform=lambda x: x[-3:-1],
    terminal_eps=v['terminal_eps'],
    distance_metric=v['distance_metric'],
    extend_dist_rew=v['extend_dist_rew'],
    only_feasible=v['only_feasible'],
    terminate_env=True,
    )


    for i_episode in range(200):
        observation = env.reset()
        print(env.init_qpos)
        print(env.model.data.qpos)
        print(env.model.data.qpos.flat[-2:].reshape(-1))
        print(env.current_goal)
        for t in range(500):
            env.render()
            action = env.action_space.sample()    # take a random action
            observation, reward, done, info = env.step(action)
        # print("REWARD: ",reward)
        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
        #     break
    # env.close()