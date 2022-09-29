# from model_functions.old_standard_functions import *
from scipy.optimize import minimize
import numpy as np
    
class DecisionModelPrototype(object):
    
    def __init__(self):
        # Dictionary of all parameters, with descriptions
        self.params = {
            # E.g.
            # 'alpha': 'Learning rate'
        }
        self.num_params = len(self.params.keys())
        # Initialize the fit_metrics
        self.fit_metrics = None

    
    # DETERMINE LIKELIHOOD OF PARAMETER
    def _llik(self, x, *args):
        raise NotImplementedError
    
    # DISCOVER PARAMETER BY MAXIMIZING LOG LIKELIHOOD
    def fit(self, actions=None, rewards=None, x0=None):
        """Use maximum likelihood to find the parameters that best explain the actions and rewards

        Args:
            actions (arr, optional): List of actions (0 or 1). Defaults to None.
            rewards (arr, optional): List of rewards (0, 0.2, or 1). Defaults to None.
            x0 (tuple, optional): Initial guesses for parameters. Defaults to None.

        Returns:
            tuple: The parameter estimates, BIC scores, and calculated values
        """
        # Maximum likelihood calculation
        optimize = True
        result = minimize(self._llik, x0, args=(actions, rewards, optimize), method='Nelder-Mead')
        param_estimate = result.x

        # Recalculate values for the best parameters
        optimize = False
        values = self._llik(param_estimate, actions, rewards, optimize)

        # Calculate BIC Score
        bic = np.log(len(actions)) * self.num_params + 2 * result.fun

        param_estimate_dict = {}
        for i, param in enumerate(self.params.keys()):
            param_estimate_dict[param] = param_estimate[i]
        return param_estimate_dict, bic, values
    
    def fit_all(self, act_rew_rate, concat=False, skip_prediction_error=False):
        """Use maximum likelihood to fit the full dataset of actions and rewards

        Args:
            act_rew_rate (structure): Contains all participants' actions and rewards for both blocks

        Returns:
            structure: Fit metrics that include parameter estimates, BIC, values and prediction errors for all participants
        """
        
        self.act_rew_rate = act_rew_rate

        self.fit_metrics = {}
        self.fit_metrics['num_participants'] = act_rew_rate['positive']['actions'].shape[0]

        if not concat:
            for block in ['positive', 'negative', 'mixed']:
                self.fit_metrics[block] = {}
                self.fit_metrics[block]['params'] = {}
                for param in self.params.keys():
                    self.fit_metrics[block]['params'][param] = []
                self.fit_metrics[block]['bic'] = []
                self.fit_metrics[block]['values'] = []
                self.fit_metrics[block]['prederr'] = []

            for block in ['positive', 'negative', 'mixed']:
                for actions, rewards in zip(act_rew_rate[block]['actions'], act_rew_rate[block]['rewards']):
                    x0 = [0.5]*self.num_params
                    param_estimate_dict, bic, values = self.fit(actions=actions, rewards=rewards, x0=x0)
                    for param, estimate in param_estimate_dict.items():
                        self.fit_metrics[block]['params'][param].append(estimate)
                    self.fit_metrics[block]['values'].append(values)
                    self.fit_metrics[block]['bic'].append(bic)

                if not skip_prediction_error:
                    self.fit_metrics[block]['prederr'] = np.zeros(act_rew_rate[block]['rewards'].shape)
                    for i, (values, choices, rewards) in enumerate(zip(self.fit_metrics[block]['values'], 
                            act_rew_rate[block]['actions'], act_rew_rate[block]['rewards'])):
                        for j, (trial_values, choice, reward) in enumerate(zip(values, choices, rewards)):
                            self.fit_metrics[block]['prederr'][i][j] = reward - trial_values[choice]
            
        elif concat:
            self.fit_metrics['combined'] = {}
            self.fit_metrics['combined']['params'] = {}
            for param in self.params.keys():
                self.fit_metrics['combined']['params'][param] = []
            self.fit_metrics['combined']['bic'] = []
            self.fit_metrics['combined']['values'] = []
            self.fit_metrics['combined']['prederr'] = []

            for i, _ in enumerate(act_rew_rate['pids']):
                actions = np.hstack([act_rew_rate[block]['actions'][i] for block in ['positive', 'negative', 'mixed']])
                rewards = np.hstack([act_rew_rate[block]['rewards'][i] for block in ['positive', 'negative', 'mixed']])
            
                x0 = [0.5]*self.num_params
                param_estimate_dict, bic, values = self.fit(actions=actions, rewards=rewards, x0=x0)
                for param, estimate in param_estimate_dict.items():
                    self.fit_metrics['combined']['params'][param].append(estimate)
                self.fit_metrics['combined']['values'].append(values)
                self.fit_metrics['combined']['bic'].append(bic)

            if not skip_prediction_error:
                self.fit_metrics['combined']['prederr'] = np.zeros(np.array(self.fit_metrics['combined']['values']).shape)
                for i, _ in enumerate(act_rew_rate['pids']):
                    actions = np.hstack([act_rew_rate[block]['actions'][i] for block in ['positive', 'negative', 'mixed']])
                    rewards = np.hstack([act_rew_rate[block]['rewards'][i] for block in ['positive', 'negative', 'mixed']])
                    values = self.fit_metrics['combined']['values']
                    for j, (trial_values, choice, reward) in enumerate(zip(values, actions, rewards)):
                        self.fit_metrics['combined']['prederr'][i][j] = reward - trial_values[choice]
        return self.fit_metrics

class SimPrototype(object):

    def __init__(self):
        self.param_set = None
        self.internal_params = None

    def _choose(self, p):
        cumulative = np.concatenate(([0], np.cumsum(p)))
        return np.max(np.nonzero(cumulative < np.random.uniform()))

    def _update_values(self, trial_num, block, values, actions, rewards, param_set, internal_params):
        raise NotImplementedError

    def _select_action(self, trial_num, block, values, actions, param_set):
        raise NotImplementedError

    def _gen_reward(self, trial_num, block, reward_structure, actions, rewards):
        """Generate a reward based on reward structure

        Args:
            trial_num (int): Current trial
            block (string): Money or other block
            reward_structure (structure): A dictionary of reward probabilities
            actions (arr): Array of actions
            rewards (arr): Array of rewards

        Returns:
            arr: Updated rewards array
        """
        trial_action = int(actions[trial_num])
        trial_success_prob = reward_structure[block][trial_action, trial_num]
        reward = np.random.uniform() < trial_success_prob
        if reward:
            reward_num = 1
        else:
            reward_num = 0
        # if reward:
        #     if np.random.uniform()<0.5:
        #         reward_num = 0.2
        #     else:
        #         reward_num = 1.0
        # else:
        #     reward_num = 0
        rewards[trial_num] = reward_num
        return rewards

    def choose_param_set(self, fitted_model=None, participant_num=None, manual_param_set=None):
        """Either use parameter set from a fitted model and a particular participant, or manually input

        Args:
            fitted_model (structure, optional): Fitted model from DecisionModels. Defaults to None.
            participant_num (int, optional): Participant number in list. Defaults to None.
            manual_param_set (dict, optional): Manual entry of parameters. Defaults to None.
        """
        if fitted_model:
            self.params = fitted_model.params
            self.param_set = {}
            for block in ['money', 'other']:
                self.param_set[block] = {}
                for param in self.params.keys():
                    self.param_set[block][param] = fitted_model.fit_metrics[block]['params'][param][participant_num]
        elif manual_param_set:
            self.param_set = manual_param_set

    def simulate(self, reward_structure, num_trials=40, simulations=100,  set_high_beta=False):
        """Use the selected parameter set to simulate data

        Args:
            reward_structure (dict): Description of reward probabilities.
            num_trials (int, optional): Total number of trials to simulate. Defaults to 40.
            simulations (int, optional): Number of participants to simulate. Defaults to 100.
            set_high_beta (bool, optional): Artificially set the beta high for softmax. Defaults to False.

        Raises:
            NotImplementedError: Should fail if choose_param_set function has not been run.

        Returns:
            dict: Return a dictionary of simulated actions and rewards, and the chosen parameter set
        """
        if not self.param_set:
            raise NotImplementedError

        if set_high_beta:
            beta = 10
        else:
            beta = None

        self.act_rew_rate = {}
        self.act_rew_rate['pids'] = np.arange(simulations)
        self.values = {}
        for block in ['money', 'other']:
            self.act_rew_rate[block] = {}
            self.values[block] = np.zeros((simulations, num_trials, 2))
            for elem in ['actions', 'rewards']:
                self.act_rew_rate[block][elem] = np.zeros((simulations, num_trials))

        for block in ['money', 'other']:
            for s_num in np.arange(simulations):
                values = np.zeros((num_trials, 2))
                values[0, :] = 0.5
                actions = np.zeros(num_trials)
                rewards = np.zeros(num_trials)
                for trial_num in np.arange(num_trials):
                    actions = self._select_action(trial_num, block, values, actions, self.param_set, beta)
                    rewards = self._gen_reward(trial_num, block, reward_structure, actions, rewards)
                    values, self.internal_params = self._update_values(trial_num, block, values, actions, rewards, self.param_set, self.internal_params)
            
                self.act_rew_rate[block]['actions'][s_num, :] = actions
                self.act_rew_rate[block]['rewards'][s_num, :] = rewards
                self.values[block][s_num, :, :] = values
            
            self.act_rew_rate[block]['actions'] = self.act_rew_rate[block]['actions'].astype(int)
        return self.act_rew_rate, self.param_set, self.values