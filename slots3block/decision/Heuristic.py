import importlib

# from model_functions.old_standard_functions import *
import numpy as np
from scipy.special import logsumexp
from slots3block.decision import Prototypes

importlib.reload(Prototypes)


class NWSLS(Prototypes.DecisionModelPrototype):
    def __init__(self):
        super().__init__()
        self.params = {
            "epsilon": "Noise parameter",
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

        epsilon = x
        if epsilon > 1 or epsilon < 0:
            return 1000

        actions, rewards, optim = args

        log_prob_actions = np.zeros(len(actions))
        probs = np.array([0.5, 0.5])

        for t, (a, r) in enumerate(zip(actions, rewards)):
            # Find log probability of observed action
            log_prob_action = np.log(probs)
            # Store the log probability of the observed action
            log_prob_actions[t] = log_prob_action[a]

            # Update probs
            if r == 1:
                probs[a] = 1 - epsilon
                probs[1 - a] = epsilon
            elif r == 0:
                probs[a] = epsilon
                probs[1 - a] = 1 - epsilon

        if optim:
            return -np.sum(log_prob_actions[1:])
        else:
            return np.zeros(len(log_prob_actions))
