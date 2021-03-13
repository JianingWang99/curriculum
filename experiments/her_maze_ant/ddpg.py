from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import special
from rllab.misc import ext
from rllab.sampler import parallel_sampler
from rllab.plotter import plotter
from functools import partial
import rllab.misc.logger as logger
import theano.tensor as TT
import pickle as pickle
import numpy as np
import pyprind
import lasagne

from curriculum.experiments.her.her_sample import her_sampler
from curriculum.experiments.her.replay_buffer import replay_buffer
from curriculum.experiments.her.normalizer import normalizer
import timeit
import csv
import time

def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **ext.compact(kwargs))
    else:
        raise NotImplementedError

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf):

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    
    bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter

#this method is handcrafted for evaluation purposes only.
def generate_goals(goals_l):
    
    inside_box_1 = bounding_box(goals_l, min_x=-1, max_x=5, min_y=-1, max_y=1)
    
    inside_box_2 = bounding_box(goals_l, min_x=3, max_x=5, min_y=-1, max_y=5)

    inside_box_3 = bounding_box(goals_l, min_x=-1, max_x=5, min_y=3, max_y=5)
        
    points_inside_box = np.concatenate((goals_l[inside_box_1], goals_l[inside_box_2], goals_l[inside_box_3]))

    return points_inside_box
        
# goid(self.env, sample_policy, self.es, goal_, itr)
def goid(env, policy, es, goal, itr, min_rew=0.1, max_rew=0.9, n_traj=3):
    
    #we exclude the final goal
    total_rew = 0
    goal_co =  goal #np.array([0, 1.2])
    for e in range(n_traj):
        # Execute policy
        observation = env.reset() #initialize observation with new goal
        es.reset()
        policy.reset()
        terminal = False
        for t in range(500):
            if terminal:
                break
            action = es.get_action(itr, observation, policy)
            observation_new, _, _, info = env.step(action)
            achieved_goal = info['achieved_goal']
            reward = env.compute_reward(achieved_goal, goal_co) + 1
            terminal = True if reward==1 else False
            total_rew += reward
    #-----GOID------
    avg_rew = total_rew/n_traj
    if min_rew <= avg_rew <= max_rew:
        return 1
    if avg_rew >= max_rew:
        return 0
    if avg_rew <= min_rew:
        return 2
    return 'error'
        
        
class DDPG(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            es,
            batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.98,
            max_path_length=250,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            eval_samples=10000,
            soft_target=True,
            soft_target_tau=0.001, #0.001
            n_updates_per_sample=1,
            scale_reward=1.0,
            include_horizon_terminal_transitions=False,
            plot=False,
            pause_for_plot=False,
            env_params=None,
            cycles=None,
            episode_number=None,
            params_action = None,
            clip_range = 5,
            replay_k=4,
            goal_center=None,
            goal_range = None,
            goal_size=None,
            seed = None):
        """
        :param env: Environment
        :param policy: Policy
        :param qf: Q function
        :param es: Exploration strategy
        :param batch_size: Number of samples for each minibatch.
        :param n_epochs: Number of epochs. Policy will be evaluated after each epoch.
        :param epoch_length: How many timesteps for each epoch.
        :param min_pool_size: Minimum size of the pool to start training.
        :param replay_pool_size: Size of the experience replay pool.
        :param discount: Discount factor for the cumulative return.
        :param max_path_length: Discount factor for the cumulative return.
        :param qf_weight_decay: Weight decay factor for parameters of the Q function.
        :param qf_update_method: Online optimization method for training Q function.
        :param qf_learning_rate: Learning rate for training Q function.
        :param policy_weight_decay: Weight decay factor for parameters of the policy.
        :param policy_update_method: Online optimization method for training the policy.
        :param policy_learning_rate: Learning rate for training the policy.
        :param eval_samples: Number of samples (timesteps) for evaluating the policy.
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained
        :param scale_reward: The scaling factor applied to the rewards when training
        :param include_horizon_terminal_transitions: whether to include transitions with terminal=True because the
        horizon was reached. This might make the Q value back up less stable for certain tasks.
        :param plot: Whether to visualize the policy performance after each eval_interval.
        :param pause_for_plot: Whether to pause before continuing when plotting.
        :return:
        """
        self.env = env
        self.policy = policy
        self.qf = qf
        self.es = es
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.discount = discount
        self.max_path_length = max_path_length
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = \
            parse_update_method(
                qf_update_method,
                learning_rate=qf_learning_rate,
            )
        self.qf_learning_rate = qf_learning_rate
        self.policy_weight_decay = policy_weight_decay
        self.policy_update_method = \
            parse_update_method(
                policy_update_method,
                learning_rate=policy_learning_rate,
            )
        self.policy_learning_rate = policy_learning_rate
        self.eval_samples = 100#eval_samples
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions
        self.plot = plot
        self.pause_for_plot = pause_for_plot

        self.qf_loss_averages = []
        self.policy_surr_averages = []
        self.q_averages = []
        self.y_averages = []
        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0
        
        self.trach_success_rate_eval = []

        self.scale_reward = scale_reward

        self.opt_info = None
        
        self.env_params = env_params
        self.cycles = cycles
        self.episode_number = episode_number
        self.params_action = params_action
        self.soft_target = soft_target
        self.clip_range = clip_range
        self.replay_k = replay_k
        
        # her sampler
        print(self.env_params)
        self.her_module = her_sampler('future', self.replay_k, self.env.compute_reward) #make reward function in goal base environment,\
                                                                            #that computes distanceof achieve_goal and desired goal
        
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.replay_pool_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.clip_range)
        
        self.goal_center = goal_center
        self.goal_range = goal_range
        self.goal_size = goal_size
        self.seed = seed


    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)
            

            
    @overrides
    def train(self):
        # This seems like a rather sequential method

        print("SUCCESS")
        
        self.start_worker()

        self.init_opt()

        itr = 0
        path_length = 0
        path_return = 0
        terminal = False
        observation = self.env.reset()

        sample_policy = pickle.loads(pickle.dumps(self.policy))
        

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            
            goal_goid = []
            for cycl in range(self.cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for ep in range(self.episode_number):
                    print(f'->epoch: {epoch} -> cycle: {cycl} -> episode {ep}')
                    #reset components
                    sample_policy.reset()
                    self.es_path_returns.append(path_return)
                    
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation_dict = self.env.reset(reset_her=True)
                    obs = observation_dict['observation']
                    
                    ag = observation_dict['achieved_goal']
                    g = observation_dict['desired_goal']

                    # start to collect samples
                    for epoch_itr in range(self.epoch_length):
                        action = self.es.get_action(itr, obs, policy=sample_policy)
#                         action = sample_policy.get_action(obs)[0]  # qf=qf)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new                        
                        ag_new = info['achieved_goal']
                        
                        #--------------------plot-----------------------------------
#                         path_id = f'{epoch}{cycl}{ep}'
#                         with open('./her_data_results/goals_pytorch.csv','a') as fd:
#                             fd.write(f'{str(path_id)} , {ag_new[0]} , {ag_new[1]}\n')
                        #--------------------plot-----------------------------------
                        
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new

                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                
                #goid
                if cycl%10==0:
                    print("Finding goid at cycle: ",cycl ," ...")
                    for gd in pyprind.prog_bar(range(self.n_updates_per_sample)):
                        batch = self.buffer.sample((self.batch_size))
                        #check goid goals
                        for gen_g in range(0, len(batch['g']), 20):
                            #exclude final goal
                            goal_to_check = batch['g'][gen_g]
                            main_goal = np.array(list(self.env.current_goal))
                            if list(goal_to_check) == list(main_goal):
                                continue
                            goal_goid.append(goid(self.env, sample_policy, self.es, goal_to_check, itr))
                        
                    
                for update_itr in pyprind.prog_bar(range(self.n_updates_per_sample)):
                    # Train policy
#                     print('Training...')
                    batch = self.buffer.sample((self.batch_size)) #self.batch_size
                    if update_itr < self.n_updates_per_sample-1:
                        self.do_training(itr, batch)
                    else:
                       #last iteration at soft update
                        self.do_training(itr, batch, soft_target_activate = True)
                #soft update after final training
                sample_policy.set_param_values(self.policy.get_param_values())
            
            with open(f'./her_data_results/her_goid_seed_{self.seed}.csv', 'a') as fp:
                #columns: 0, 1, 2
                wr = csv.writer(fp, dialect = 'excel')
                easy = goal_goid.count(0)
                inter = goal_goid.count(1)
                hard = goal_goid.count(2)
                final_res = [easy, inter, hard]
                wr.writerow(final_res)
                
            logger.log("Training finished")
            if True:
                evalu = self.evaluate(epoch, self.buffer, pol = sample_policy)
                #save a a csv file
                print(f'success rate after eval: {evalu}')
                #write succ to file
                with open(f'./her_data_results/experiment_1_seed_{self.seed}.csv','a') as fd:
                    fd.write(f'{evalu}\n')
                    
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.env.terminate()
        self.policy.terminate()

        

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        
    def _preproc_og(self, o, g):
        o = np.clip(o, -200, 200)
        g = np.clip(g, -200, 200)
        return o, g
    
    def init_opt(self):

        # First, create "target" policy and Q functions
        target_policy = pickle.loads(pickle.dumps(self.policy))
        target_qf = pickle.loads(pickle.dumps(self.qf))

        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        
        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        
        yvar = TT.vector('ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([TT.sum(TT.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)
        
        

        qf_loss = TT.mean(TT.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([TT.sum(TT.square(param))
                                        for param in self.policy.get_params(regularizable=True)])
        
        #real qval, ovbs, real_actions
        policy_qval = self.qf.get_qval_sym(
            obs, self.policy.get_action_sym(obs),
            deterministic=True
        )
        #maybe actor loss (99.9999% sure)
        policy_surr = -TT.mean(policy_qval)

        #add value here to avoid vanishing gradient, "self.policy.get_action_sym(obs) - > gets real actions"
        policy_surr += TT.mean(TT.pow(self.policy.get_action_sym(obs), 2))


        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_updates = self.qf_update_method(
            qf_reg_loss, self.qf.get_params(trainable=True))
        policy_updates = self.policy_update_method(
            policy_reg_surr, self.policy.get_params(trainable=True))

        f_train_qf = ext.compile_function(
            inputs=[yvar, obs, action],
            outputs=[qf_loss, qval],
            updates=qf_updates
        )

        f_train_policy = ext.compile_function(
            inputs=[obs],
            outputs=policy_surr,
            updates=policy_updates
        )

        self.opt_info = dict(
            f_train_qf=f_train_qf,
            f_train_policy=f_train_policy,
            target_qf=target_qf,
            target_policy=target_policy,
        )

    def do_training(self, itr, batch, soft_target_activate = False):
                                               
        #TODO: CONSIDER NORMALIZATION AND CLIPPING
        obs, actions, rewards, next_obs, goals = ext.extract(batch,\
                                                             'obs', 'actions', 'r', 'obs_next', 'g')

        obs_t, goals_t = self._preproc_og(obs, goals)
        next_obs_t, goals_next = self._preproc_og(next_obs, goals)
        
        obs_norm = self.o_norm.normalize(obs_t)
        g_norm = self.g_norm.normalize(goals_t)
        
        obs_next_norm = self.o_norm.normalize(next_obs_t)
        g_next_norm = self.g_norm.normalize(goals_next)
        
        input_obs = obs_norm.copy()
        input_next_obs = obs_next_norm.copy()

        #-----------------concaternate observations with goals-------------------------------
        for i in range(len(obs)):
            input_obs[i][-2:] = g_norm[i]
        for i in range(len(next_obs)):
            input_next_obs[i][-2:] = g_next_norm[i]
        
        # compute the on-policy y values
        target_qf = self.opt_info["target_qf"]
        target_policy = self.opt_info["target_policy"]

        next_actions, _ = target_policy.get_actions(input_next_obs) #next_obs

        next_qvals = target_qf.get_qval(input_next_obs, next_actions).reshape(self.batch_size, 1)
        ys = rewards + self.discount * next_qvals
        ys = ys.copy().reshape(self.batch_size, )
        
        #---clipping according to HER paper---
        clip_return = -1 / (1 - self.discount)
        ys[ys < clip_return] = clip_return
        ys[ys > 0.] = 0
        #------done clipping--------

        f_train_qf = self.opt_info["f_train_qf"]
        f_train_policy = self.opt_info["f_train_policy"]

        #critic_loss (qf loss), real q_value
        qf_loss, qval = f_train_qf(ys, input_obs, actions) #ys, obs, actions
        

        #actor loss = policu_surr
        policy_surr = f_train_policy(input_obs)
        #soft update
        if soft_target_activate:
            target_policy.set_param_values(
                target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
                self.policy.get_param_values() * self.soft_target_tau)
            target_qf.set_param_values(
                target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
                self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.policy_surr_averages.append(policy_surr)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

        
    def evaluate(self, epoch, pool, pol=None):
        #TODO: USE GOAL GEN FOR EVAL
        print("SAMPLING GOAL")
        
        goals_g = np.random.uniform(np.array(self.goal_center) - np.array(self.goal_range), 
                                  np.array(self.goal_center) + np.array(self.goal_range), size=(3000, self.goal_size)) #original 300
        
        goals = generate_goals(goals_g)

        sampl_pol = pol
        total_success_rate = []
        for _ in pyprind.prog_bar(range(self.eval_samples)):
            per_success_rate = []
            observation = self.env.reset(reset_her=True)
            obs = observation['observation']
            
            #sample new goal
            new_g = goals[np.random.randint(0, len(goals)-1)]
            sampl_pol.reset()
            for _ in range(self.env_params['max_timesteps']):
                #sample_goal and conc with ovservation
                action = sampl_pol.get_action(obs)[0]  # qf=qf)
#                 print("action:")
#                 print(action)
                # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
                obs = observation_new
                ag = info['achieved_goal']
                rew = self.env.compute_reward(ag, new_g)
                #the succ is just a parameter that indicates whether the goal is achieved (1) or not (0)
                succ = 1 if rew==0 else 0
                per_success_rate.append(succ)
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)

        return total_success_rate.mean()







#         logger.log("Collecting samples for evaluation")                
#         paths = parallel_sampler.sample_paths(
#             policy_params=self.opt_info["target_policy"].get_param_values(),  #self.policy.get_param_values(), #use target
#             max_samples=self.eval_samples,
#             max_path_length=self.max_path_length,
#         ) #list of dictionaries

#         print(paths[0]['observations'].shape)
#         #dict_keys(['observations', 'actions', 'rewards', 'agent_infos', 'env_infos', 'dones', 'last_obs'])

#         exit()
#         self.trach_success_rate_eval = np.mean([path['dones'] for path in paths])

#         average_discounted_return = np.mean(
#             [special.discount_return(path["rewards"], self.discount) for path in paths]
#         )

#         returns = [sum(path["rewards"]) for path in paths]

#         all_qs = np.concatenate(self.q_averages)
#         all_ys = np.concatenate(self.y_averages)

#         average_q_loss = np.mean(self.qf_loss_averages)
#         average_policy_surr = np.mean(self.policy_surr_averages)
#         average_action = np.mean(np.square(np.concatenate(
#             [path["actions"] for path in paths]
#         )))

#         policy_reg_param_norm = np.linalg.norm(
#             self.policy.get_param_values(regularizable=True)
#         )
#         qfun_reg_param_norm = np.linalg.norm(
#             self.qf.get_param_values(regularizable=True)
#         )

#         logger.record_tabular('Epoch', epoch)
#         logger.record_tabular('AverageReturn',
#                               np.mean(returns))
#         logger.record_tabular('StdReturn',
#                               np.std(returns))
#         logger.record_tabular('MaxReturn',
#                               np.max(returns))
#         logger.record_tabular('MinReturn',
#                               np.min(returns))
#         if len(self.es_path_returns) > 0:
#             logger.record_tabular('AverageEsReturn',
#                                   np.mean(self.es_path_returns))
#             logger.record_tabular('StdEsReturn',
#                                   np.std(self.es_path_returns))
#             logger.record_tabular('MaxEsReturn',
#                                   np.max(self.es_path_returns))
#             logger.record_tabular('MinEsReturn',
#                                   np.min(self.es_path_returns))
#         logger.record_tabular('AverageDiscountedReturn',
#                               average_discounted_return)
#         logger.record_tabular('AverageQLoss', average_q_loss)
#         logger.record_tabular('AveragePolicySurr', average_policy_surr)
#         logger.record_tabular('AverageQ', np.mean(all_qs))
#         logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
#         logger.record_tabular('AverageY', np.mean(all_ys))
#         logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
#         logger.record_tabular('AverageAbsQYDiff',
#                               np.mean(np.abs(all_qs - all_ys)))
#         logger.record_tabular('AverageAction', average_action)

#         logger.record_tabular('PolicyRegParamNorm',
#                               policy_reg_param_norm)
#         logger.record_tabular('QFunRegParamNorm',
#                               qfun_reg_param_norm)

#         self.env.log_diagnostics(paths)
#         self.policy.log_diagnostics(paths)
        

        self.qf_loss_averages = []
        self.policy_surr_averages = []

        self.q_averages = []
        self.y_averages = []
        self.es_path_returns = []

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            policy=self.policy,
            target_qf=self.opt_info["target_qf"],
            target_policy=self.opt_info["target_policy"],
            es=self.es,
        )
    
    
    
    
    
    
    
    
    
    
    
    

#             sample_policy = pickle.loads(pickle.dumps(self.policy))
#         total_success_rate = []
#         print("GET PARAM VALUES")
#         print(self.policy.get_param_values())
#         exit()
#         for _ in pyprind.prog_bar(range(self.eval_samples)):
#             per_success_rate = []
#             observation = self.env.reset(reset_her=True)
#             obs = observation['observation']
#             g = observation['desired_goal']
            
#             self.es.reset()
#             sample_policy.reset()
#             for _ in range(self.env_params['max_timesteps']):
#                 action = self.es.get_action(9999, observation, policy=sample_policy)  # qf=qf)
#                 # feed the actions into the environment
#                 observation_new, _, _, info = self.env.step(action)
#                 obs = observation_new
#                 g = info['goal']
#                 per_success_rate.append(info['goal_reached'])
#             total_success_rate.append(per_success_rate)
#         total_success_rate = np.array(total_success_rate)
#         print("total success rate")
#         print(total_success_rate)
#         return total_success_rate.mean()
