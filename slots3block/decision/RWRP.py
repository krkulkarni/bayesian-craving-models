import importlib
# from model_functions.old_standard_functions import *
import numpy as np
from scipy.special import logsumexp
from slots3block.decision import Prototypes
importlib.reload(Prototypes)
    
class RWRP(Prototypes.DecisionModelPrototype):
    
    def __init__(self):
        super().__init__()
        self.params = {
            'alpha_pos': 'Learning rate for positive RPEs',
            'alpha_neg': 'Learning rate for negative RPEs',
            'beta': 'Inverse temperature for softmax'
        }
        self.num_params = len(self.params.keys())
        # Initialize the fit_metrics
        self.fit_metrics = None
    
    # DETERMINE LIKELIHOOD OF PARAMETER
    def _llik(self, x, *args):
        """Likelihood functions for RWRP model

        Args:
            x (tuple): Contains the parameters to be optimized
            *args (tuple): Contains the actions, rewards, and optimization flag

        Returns:
            neg_log_lik: negative log likelihood of the given parameters
            Qs: values calculated from the given parameters
        """
        # Unpack the parameters
        alpha_pos, alpha_neg, beta = x
        # Unpack the other arguments
        # optim - if true, returns the neg log lik
        #         if false, returns the Q values
        actions, rewards, optim = args

        # If the parameter guesses are greater than 5 or less than -5,
        # it is unlikely, so return 1000
        for param in x:
            if np.abs(param) > 20:
                return 1000
        
        if beta < 0:
            return 1000

        # Create a list with the Q values of each trial
        Qs = np.ones((len(actions), 2), dtype=float)
        # Initialize the first set of values as 0.5
        Qs[0,:] = 0.5
        # Loop through the actions and rewards to generate values
        # Stop before the last action because it doesn't matter
        for t, (a, r) in enumerate(zip(actions[:-1], rewards[:-1])):
            delta = r - Qs[t, a]
            if delta>=0:
                Qs[t+1, a] = Qs[t, a] + alpha_pos * delta
                # if Qs[t+1, a] > 1:
                #     return 1000
            elif delta<0:
                Qs[t+1, a] = Qs[t, a] + alpha_neg * delta
            Qs[t+1, 1-a] = Qs[t, 1-a]

        # Apply the softmax transformation in a vectorized way to the values
        Qs_ = Qs * beta
        log_prob_actions = Qs_ - logsumexp(Qs_, axis=1)[:, None]

        # Return the log_prob_actions for the observed actions
        log_prob_actions = log_prob_actions[np.arange(len(actions)), actions]
        if optim:
            return -np.sum(log_prob_actions[1:])
        else:
            return Qs

class RWRP_SimActRew(Prototypes.SimPrototype):
    
    def __init__(self):
        super().__init__()
        self.params = {
            'alpha_pos': 'Learning rate for positive RPEs',
            'alpha_neg': 'Learning rate for negative RPEs',
            'beta': 'Inverse temperature for softmax'
        }
        self.num_params = len(self.params.keys())
    
    
    def _update_values(self, trial_num, block, values, actions, rewards, param_set, internal_params):
        if trial_num + 2 >= len(actions):
            return values, internal_params

        trial_reward = rewards[trial_num]
        trial_action = int(actions[trial_num])
        rpe = trial_reward - values[trial_num, trial_action]
        if rpe > 0:
            values[trial_num+1, trial_action] = values[trial_num, trial_action] + param_set[block]['alpha_pos']*(rpe)
            values[trial_num+1, 1-trial_action] = values[trial_num, 1-trial_action]
        elif rpe <= 0:
            values[trial_num+1, trial_action] = values[trial_num, trial_action] + param_set[block]['alpha_neg']*(rpe)
            values[trial_num+1, 1-trial_action] = values[trial_num, 1-trial_action]  
        
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