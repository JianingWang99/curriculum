import matplotlib
import csv

matplotlib.use('Agg')
import os
import os.path as osp
import multiprocessing
import random
import numpy as np
import tensorflow as tf
import tflearn
from collections import OrderedDict

from rllab.misc import logger
from curriculum.logging import HTMLReport
from curriculum.logging import format_dict
from curriculum.logging.logger import ExperimentLogger
from curriculum.logging.visualization import plot_labeled_states

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# from curriculum.experiments.ddpg.ddpg import DDPG
from curriculum.experiments.her.ddpg import DDPG
#from curriculum.experiments.ddpg.maze_ant_her_workingUntilCorner import DDPG

from rllab.envs.normalized_env import normalize
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from curriculum.experiments.her.gaussian_strategy_her import GaussianStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from curriculum.state.evaluator import label_states
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, \
    FixedStateGenerator
from curriculum.state.generator import StateGAN
from curriculum.state.utils import StateCollection

from curriculum.envs.goal_env import generate_initial_goals
from curriculum.experiments.her.goal_env import GoalExplorationEnv
from curriculum.envs.maze.maze_evaluate import test_and_plot_policy  # TODO: make this external to maze env
from curriculum.envs.maze.maze_ant.ant_maze_env import AntMazeEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]




def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    # logger.log("Initializing report and plot_policy_reward...")
    # log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    # report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    # report.add_header("{}".format(EXPERIMENT_TYPE))
    # report.add_text(format_dict(v))

    inner_env = normalize(AntMazeEnv(maze_id=v['maze_id']))
    env = inner_env


    uniform_goal_generator = UniformStateGenerator(state_size=v['goal_size'], bounds=v['goal_range'],
                                                   center=v['goal_center'])
    
    uniform_goal_generator = FixedStateGenerator(v['final_goal'])
    
    env = GoalExplorationEnv(
        env=inner_env, goal_generator=uniform_goal_generator,
        obs2goal_transform=lambda x: x[-3:-1],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        only_feasible=v['only_feasible'],
        append_goal_to_observation=True,
        terminate_env=True
    )


    #for ddpg: exploration strategy
    es = GaussianStrategy(env_spec = env.spec, params_action = v['params_action'], env_params = v['env_params'])
    #critic network
    qf = ContinuousMLPQFunction(
        env_spec=env.spec,
        hidden_sizes = (64, 64))
    
    print("before policy initialization:")
    policy = DeterministicMLPPolicy( 
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        # Fix the variance since different goals will require different variances, making this parameter hard to learn.
        #learn_std=v['learn_std'],
        #adaptive_std=v['adaptive_std'],
        #std_hidden_sizes=(16, 16),  # this is only used if adaptive_std is true!
        #output_gain=v['output_gain'],
        #init_std=v['policy_init_std'],
    )

    

        
    env.update_goal_generator(FixedStateGenerator(v['final_goal']))
    
    logger.log("Training the algorithm")
    
    algo = DDPG(
        env=env,
        es = es,
        qf = qf,   
        policy=policy,
        batch_size=v['pg_batch_size'],
        max_path_length=v['horizon'],
        n_epochs=v['inner_iters'],
        env_params=v['env_params'],
        cycles=v['cycles'],
        episode_number=v['episode_number'],
        epoch_length = v['epoch_length'],
        n_updates_per_sample = v['n_updates_per_sample'],
        params_action = v['params_action'],
        soft_target_tau = v['polyak'],
        clip_range = v['clip_range'],
        #for goal gen
        goal_center = v['goal_center'],
        goal_range = v['goal_range'],
        goal_size = v['goal_size'],
        seed = v['seed']
    )
    
    algo.train()
    