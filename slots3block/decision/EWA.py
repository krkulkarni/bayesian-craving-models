import importlib
# from model_functions.old_standard_functions import *
import numpy as np
from scipy.special import logsumexp
from slots3block.decision import Prototypes
importlib.reload(Prototypes)
    
class EWA(Prototypes.DecisionModelPrototype):
    
    def __init__(self, constrained=20):
        super().__init__()
        self.params = {
            'phi': 'Decay factor for previous payoffs (analogous to learning rate in RW)',
            'rho': 'Experience decay factor',
            'beta': 'Inverse temperature for softmax'
        }
        self.num_params = len(self.params.keys())
        # Initialize the fit_metrics
        self.fit_metrics = None
        self.constrained = constrained
    
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
        phi, rho, beta = x
        actions, rewards, optimize = args
        
        if self.constrained:
            for param in x:
                if np.abs(param) > self.constrained:
                    return 1000
        
            if beta < 0:
                return 1000

        # Initialize experience weights (eta)
        eta_prev = np.array([1,1])
        eta_now = np.array([1,1])
        
        # Create a list with the Q values of each trial
        Qs = np.ones((len(actions), 2), dtype=float)
        Qs[0,:] = 0.5
        try:
            for t, (a, r) in enumerate(zip(actions[:-1], rewards[:-1])):  # The last Q values were never used, so there is no need to compute them
                eta_now[a] = eta_prev[a]*rho+1
                Qs[t+1, a] = (Qs[t, a] * phi  * eta_prev[a] + r) / eta_now[a]
                Qs[t+1, 1-a] = Qs[t, 1-a]
                eta_prev = eta_now
        except OverflowError:
            return 1000

        # Apply the softmax transformation in a vectorized way
        Qs_ = Qs * beta
        log_prob_actions = Qs_ - logsumexp(Qs_, axis=1)[:, None]

        # Return the log_prob_actions for the observed actions
        log_prob_actions = log_prob_actions[np.arange(len(actions)), actions]
        if optimize:
            return -np.sum(log_prob_actions[1:])
        else:
            return Qs

class EWA_SimActRew(Prototypes.SimPrototype):
    
    def __init__(self):
        super().__init__()
        self.params = {
            'phi': 'Decay factor for previous payoffs (analogous to learning rate in RW)',
            'rho': 'Experience decay factor',
            'beta': 'Inverse temperature for softmax'
        }
        self.internal_params = {
            'money': {
                'eta_now': np.array([1,1]),
                'eta_prev': np.array([1,1])
            },
            'other': {
                'eta_now': np.array([1,1]),
                'eta_prev': np.array([1,1])
            }
        }
        self.num_params = len(self.params.keys())
    
    
    def _update_values(self, trial_num, block, values, actions, rewards, param_set, internal_params):
        if trial_num + 2 >= len(actions):
            return values, internal_params

        trial_reward = rewards[trial_num]
        trial_action = int(actions[trial_num])

        internal_params[block]['eta_now'][trial_action] = internal_params[block]['eta_prev'][trial_action]*param_set[block]['rho']+1
        values[trial_num+1, trial_action] = (values[trial_num, trial_action] * param_set[block]['phi']  * internal_params[block]['eta_prev'][trial_action] + trial_reward) / internal_params[block]['eta_now'][trial_action]
        values[trial_num+1, 1-trial_action] = values[trial_num, 1-trial_action]
        internal_params[block]['eta_prev'] = internal_params[block]['eta_now']
        
        return values, internal_params
    
    
    def _select_action(self, trial_num, block, values, actions, param_set, beta=None):
        trial_value = values[trial_num, :]
        if beta:
            chosen_beta = beta
        else:
            chosen_beta = param_set[block]['beta']
        p = np.exp(chosen_beta*trial_value)/np.sum(np.exp(chosen_beta*trial_value))
        actions[trial_num] = self._choose(p)
        return actions