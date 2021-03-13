import os
import random

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import tflearn
import argparse
import sys
from multiprocessing import cpu_count
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator
from rllab import config

from curriculum.experiments.her_free_maze.maze_ant_ddpg_algo import run_task

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ec2', '-e', action='store_true', default=False, help="add flag to run in ec2")
    parser.add_argument('--clone', '-c', action='store_true', default=False,
                        help="add flag to copy file and checkout current")
    parser.add_argument('--local_docker', '-d', action='store_true', default=False,
                        help="add flag to run in local dock")
    parser.add_argument('--type', '-t', type=str, default='', help='set instance type')
    parser.add_argument('--price', '-p', type=str, default='', help='set betting price')
    parser.add_argument('--subnet', '-sn', type=str, default='', help='set subnet like us-west-1a')
    parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
    parser.add_argument('--debug', action='store_true', default=False, help="run code without multiprocessing")
    args = parser.parse_args()

    # setup ec2
    subnets = [
        'ap-south-1a', 'us-east-2c', 'us-east-2b', 'us-east-2a', 'ap-northeast-2a', 'ap-northeast-2c', 'ap-south-1b'
    ]
    ec2_instance = args.type if args.type else 'c4.4xlarge'
    # configure instan
    info = config.INSTANCE_TYPE_INFO[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"])
    n_parallel = int(info["vCPU"] / 2)  # make the default 4 if not using ec2
    if args.ec2:
        mode = 'ec2'
    elif args.local_docker:
        mode = 'local_docker'
        n_parallel = cpu_count() if not args.debug else 1
    else:
        mode = 'local'
        n_parallel = cpu_count() if not args.debug else 1
        # n_parallel = multiprocessing.cpu_count()
    n_parallel = 8
    exp_prefix = 'her-maze-ant'

    vg = VariantGenerator()
    vg.add('goal_size', [2])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('terminal_eps', [1]) #og 1
    vg.add('only_feasible', [False])
#     vg.add('maze_id', [0])
    vg.add('goal_range', [5])  # this will be used also as bound of the state_space
    vg.add('goal_center', [(0, 0)])
    # goal-algo params
    vg.add('min_reward', [0])
    vg.add('max_reward', [1])
    vg.add('distance_metric', ['L2'])
    vg.add('extend_dist_rew', [1])  # !!!!
    vg.add('persistence', [1])
    vg.add('n_traj', [3])  # only for labeling and plotting (for now, later it will have to be equal to persistence!)
    vg.add('with_replacement', [False])

    vg.add('unif_goals', [False])  # put False for fixing the goal below!
    vg.add('final_goal', [(0, 4)]) #0,4 og

    # sampling params
    vg.add('horizon', [500]) #original: 500
#     vg.add('outer_iters', lambda maze_id: [500])
    vg.add('inner_iters', [200]) #change for ddpg original:5
    vg.add('pg_batch_size', [128]) #change for ddpg original:100000
    # policy initialization
    vg.add('output_gain', [1])
    vg.add('policy_init_std', [1])
    vg.add('learn_std', [True])
    vg.add('adaptive_std', [False])
    vg.add('env_params', [{'obs': 133, 'goal': 2, 'action': 8, 'action_max': 1.0, 'max_timesteps': 500}])
    vg.add('cycles',[50])
    vg.add('episode_number',[16])
    vg.add('epoch_length', [500])
    vg.add('n_updates_per_sample', [40])  
    vg.add('params_action', [{'noise_eps':0.2 , 'random_eps': 0.3}])
    vg.add('polyak',[0.95]) #0.95
    vg.add('clip_range', [5])
    vg.add('seed', range(500, 520, 20))
#     vg.add('seed', range(420, 520, 20))

#     vg.add('seed', [500])


    # Launching
    print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_prefix, vg.size))
    print('Running on type {}, with price {}, parallel {} on the subnets: '.format(config.AWS_INSTANCE_TYPE,
                                                                                   config.AWS_SPOT_PRICE, n_parallel),
          *subnets)

    for vv in vg.variants():
        run_experiment_lite(
            # use_cloudpickle=False,
            stub_method_call=run_task,
            log_dir = "./HER/",
            variant=vv,
            mode='local',
            n_parallel=1,# 8, #n_parallel, #,1
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=vv['seed'],
            exp_prefix=exp_prefix,
    #          plot = True,
            # exp_name=exp_name,
        )
