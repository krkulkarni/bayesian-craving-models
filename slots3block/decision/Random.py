import importlib
# from model_functions.old_standard_functions import *
import numpy as np
from scipy.special import logsumexp
from slots3block.decision import Prototypes
importlib.reload(Prototypes)
    
class Random(Prototypes.DecisionModelPrototype):
    
    def __init__(self):
        super().__init__()
        self.params = {
            'bias': 'Bias for one side',
        }
        self.num_params = len(self.params.keys())
        # Initialize the fit_metrics
        self.fit_metrics = None
    
    # DETERMINE LIKELIHOOD OF PARAMETER
    def _llik(self, x, *args):
        """Likelihood functions for biased model

        Args:
            x (tuple): Contains the parameters to be optimized
            *args (tuple): Contains the actions, rewards, and optimization flag

        Returns:
            neg_log_lik: negative log likelihood of the given parameters
            Qs: values calculated from the given parameters
        """

        b = x
        if b>1:
            return 1000
        
        actions, rewards, optim = args

        log_prob_actions = np.zeros(len(actions))

        for t, (a, r) in enumerate(zip(actions, rewards)):
            # Find log probability of observed action
            log_prob_action = np.log([b, 1-b])

            # Store the log probability of the observed action
            log_prob_actions[t] = log_prob_action[a]

        if optim:
            return -np.sum(log_prob_actions[1:])
        else:
            return np.zeros(len(log_prob_actions))

class Random_SimActRew(Prototypes.SimPrototype):
    
    def __init__(self):
        super().__init__()
        self.params = {
            'bias': 'Bias for one side',
        }
        self.num_params = len(self.params.keys())
    
    
    def _update_values(self, trial_num, block, values, actions, rewards, param_set, internal_params):
        return values, None
    
    
    def _select_action(self, trial_num, block, values, actions, param_set, beta=None):
        actions[trial_num] = self._choose(param_set[block]['bias'])
        return actions