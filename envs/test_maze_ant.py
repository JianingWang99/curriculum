from curriculum.envs.maze.maze_ant.ant_maze_env import AntMazeEnv

env = AntMazeEnv()

for i_episode in range(200):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()    # take a random action
        observation, reward, done, info = env.step(action)
        # print("REWARD: ",reward)
        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
        #     break
env.close()
