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

from curriculum.experiments.her.algo_her import HER, SimpleReplayPool
from rllab.envs.normalized_env import normalize
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from curriculum.experiments.her.her_evaluator import label_states, label_states_from_paths
from curriculum.envs.base import UniformListStateGenerator, UniformStateGenerator, \
    FixedStateGenerator
from curriculum.state.generator import StateGAN
from curriculum.state.utils import StateCollection

from curriculum.envs.goal_env import GoalExplorationEnv, generate_initial_goals
from curriculum.experiments.ddpg.ddpg_maze_evaluate import test_and_plot_policy, sample_unif_feas  # TODO: make this external to maze env
from curriculum.envs.maze.maze_ant.ant_maze_env import AntMazeEnv

EXPERIMENT_TYPE = osp.basename(__file__).split('.')[0]


def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    sampling_res = 2 if 'sampling_res' not in v.keys() else v['sampling_res']
    samples_per_cell = 10

    # Log performance of randomly initialized policy with FIXED goal [0.1, 0.1]
    logger.log("Initializing report and plot_policy_reward...")
    log_dir = logger.get_snapshot_dir()  # problem with logger module here!!
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=3)

    report.add_header("{}".format(EXPERIMENT_TYPE))
    report.add_text(format_dict(v))

    # tf_session = tf.Session()

    inner_env = normalize(AntMazeEnv(maze_id=v['maze_id']))

    # fixed_goal_generator = FixedStateGenerator(v['final_goal'])
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


    #for ddpg: exploration strategy
    es = OUStrategy(env_spec = env.spec)
    #critic network
    qf = ContinuousMLPQFunction(env_spec=env.spec)

    pool = SimpleReplayPool(
                max_pool_size=v['ddpg_replay_pool_size'],
                observation_dim=env.observation_space.flat_dim,
                action_dim=env.action_space.flat_dim,
            )

    policy = DeterministicMLPPolicy( 
        env_spec=env.spec,
        hidden_sizes=(64, 64),
    )

    # initialize all logging arrays on itr0
    outer_iter = 0

    logger.log('Generating the Initial Heatmap...')
    test_and_plot_policy(policy, es, env, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                         itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'])
    report.new_row()

    for outer_iter in range(1, v['outer_iters']):
        logger.log("Outer itr # %i" % outer_iter)

        goals = sample_unif_feas(env, samples_per_cell=samples_per_cell)

        with ExperimentLogger(log_dir, 'last', snapshot_mode='last', hold_outter_log=True):
            logger.log("Updating the environment goal generator")
            if v['unif_goals']:
                env.update_goal_generator(
                    UniformListStateGenerator(
                        goals.tolist(), persistence=v['persistence'], with_replacement=v['with_replacement'],
                    )
                )
            else:
                env.update_goal_generator(FixedStateGenerator(v['final_goal']))

            logger.log("Training the algorithm")

            print("ddpg policy param value before: ", policy.get_param_values())
            print("ddpg pool size, ", pool.size)
            
            algo = HER(
                v=v,
                env=env,
                es = es,
                qf = qf,   
                policy=policy,
                pool=pool,
                batch_size=v['ddpg_batch_size'],
                max_path_length=v['horizon'],
                n_epochs=v['inner_iters'],
                epoch_length=v['horizon'],
                min_pool_size=v['ddpg_min_pool_size'],
                replay_pool_size=v['ddpg_replay_pool_size'],
                plot = False,
            )
            inner_begin_iter = (outer_iter-1)*v['inner_iters']+1
            algo.train(itr=inner_begin_iter)

            print("ddpg policy param value after: ", policy.get_param_values())
            print("ddpg pool size, ", pool.size)


        logger.log('Generating the Heatmap...')
        test_and_plot_policy(policy, es, env, max_reward=v['max_reward'], sampling_res=sampling_res, n_traj=v['n_traj'],
                             itr=outer_iter, report=report, limit=v['goal_range'], center=v['goal_center'])

        logger.log("Labeling the goals")
        labels = label_states(goals, env, policy, es, v['horizon'], n_traj=v['n_traj'], key='goal_reached')

        plot_labeled_states(goals, labels, report=report, itr=outer_iter, limit=v['goal_range'],
                            center=v['goal_center'], maze_id=v['maze_id'])

        # ###### extra for deterministic:
        # logger.log("Labeling the goals deterministic")
        # with policy.set_std_to_0():
        #     labels_det = label_states(goals, env, policy, v['horizon'], n_traj=v['n_traj'], n_processes=1)
        # plot_labeled_states(goals, labels_det, report=report, itr=outer_iter, limit=v['goal_range'], center=v['goal_center'])

        logger.dump_tabular(with_prefix=False)
        report.new_row()
