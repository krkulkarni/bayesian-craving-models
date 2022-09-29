import importlib
# from model_functions.old_standard_functions import *
import numpy as np
from scipy.special import logsumexp
from slots3block.decision import Prototypes
importlib.reload(Prototypes)
    
class FictiveRW(Prototypes.DecisionModelPrototype):
    
    def __init__(self):
        super().__init__()
        self.params = {
            'alpha': 'Learning rate for chosen option',
            'fictive_alpha': 'Learning rate for unchosen option',
            'beta': 'Inverse temperature for softmax'
        }
        self.num_params = len(self.params.keys())
        # Initialize the fit_metrics
        self.fit_metrics = None
    
    # DETERMINE LIKELIHOOD OF PARAMETER
    def _llik(self, x, *args):
        """Likelihood functions for RW model

        Args:
            x (tuple): Contains the parameters to be optimized
            *args (tuple): Contains the actions, rewards, and optimization flag

        Returns:
            neg_log_lik: negative log likelihood of the given parameters
            Qs: values calculated from the given parameters
        """
        alpha, fictive_alpha, beta = x
        actions, rewards, optim = args

        for param in x:
            if np.abs(param) > 20:
                return 1000
        
        if beta < 0:
            return 1000

        Qs = np.ones((len(actions), 2), dtype=float)
        Qs[0,:] = 0.5
        for t, (a, r) in enumerate(zip(actions[:-1], rewards[:-1])):
            delta = r - Qs[t, a]
            fictive_delta = (1-r) - Qs[t, 1-a]
            Qs[t+1, a] = Qs[t, a] + alpha * delta
            Qs[t+1, 1-a] = Qs[t, 1-a] + fictive_alpha * fictive_delta

        # Apply the softmax transformation in a vectorized way to the values
        Qs_ = Qs * beta
        log_prob_actions = Qs_ - logsumexp(Qs_, axis=1)[:, None]

        # Return the log_prob_actions for the observed actions
        log_prob_actions = log_prob_actions[np.arange(len(actions)), actions]
        if optim:
            return -np.sum(log_prob_actions[1:])
        else:
            return Qs

class FictiveRW_SimActRew(Prototypes.SimPrototype):
    
    def __init__(self):
        super().__init__()
        self.params = {
            'alpha': 'Learning rate for chosen option',
            'fictive_alpha': 'Learning rate for unchosen option',
            'beta': 'Inverse temperature for softmax'
        }
        self.num_params = len(self.params.keys())
    
    
    def _update_values(self, trial_num, block, values, actions, rewards, param_set, internal_params):
        if trial_num + 2 >= len(actions):
            return values, None

        trial_reward = rewards[trial_num]
        trial_action = int(actions[trial_num])
        rpe = trial_reward - values[trial_num, trial_action]
        fictive_rpe = (1 - trial_reward) - values[trial_num, 1-trial_action]
        values[trial_num+1, trial_action] = values[trial_num, trial_action] + param_set[block]['alpha']*(rpe)
        values[trial_num+1, 1-trial_action] = values[trial_num, 1-trial_action] + param_set[block]['fictive_alpha']*(fictive_rpe)
        
        return values, None
    
    
    def _select_action(self, trial_num, block, values, actions, param_set, beta=None):
        trial_value = values[trial_num, :]
        if beta:
            chosen_beta = beta
        else:
            chosen_beta = param_set[block]['beta']
        p = np.exp(chosen_beta*trial_value)/np.sum(np.exp(chosen_beta*trial_value))
        actions[trial_num] = self._choose(p)
        return actions