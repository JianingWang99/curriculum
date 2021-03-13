from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np


class GaussianStrategy(ExplorationStrategy, Serializable):
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.
    """

    def __init__(self, env_spec, params_action, env_params):
        assert isinstance(env_spec.action_space, Box)
        assert len(env_spec.action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self.params_action = params_action
        self.env_params = env_params


    def get_action(self, t, observation, policy, **kwargs):
        action = policy.get_action(observation)[0]
        action += self.params_action['noise_eps'] * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])

        action += np.random.binomial(1, self.params_action['random_eps'], 1)[0] * (random_actions - action)
        return action
