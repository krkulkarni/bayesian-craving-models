## Latest version of the models for the slotscraving task
## Author: Kaustubh Kulkarni
## Original Date: June 2022
## Last Modified: Aug 9, 2022

import numpy as np
# import pandas as pd
import pymc as pm
import aesara.tensor as at
import aesara
from scipy import stats

from abc import ABC, abstractmethod

from sys import path
import os
from IPython.display import clear_output

import arviz as az

## Mixed EVRPE-CEC prototype class
class MixedPrototype(ABC):
    def __init__(self, longform, summary, project_dir, save_path):
        self.name = None        
        self.longform = longform
        self.summary = summary
        self.pid_list = longform['PID'].unique()
        self.traces = {
            'money': {},
            'other': {}
        }
        self.project_dir = project_dir

        num_craving_trials = 20
        num_blocks = 2
        self.craving_inds = None
        self.mean_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.std_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.norm_cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self.cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self._calc_norm_cravings()

        self.save_path = save_path
    
    def _calc_norm_cravings(self):
        for pid_num in range(len(self.pid_list)):
            for b, block in enumerate(['money', 'other']):
                pid = self.pid_list[pid_num]
                cravings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values
                craving_inds = np.squeeze(np.argwhere(cravings>-1))
                mask = np.ones(len(craving_inds), dtype=bool)
                mask[12] = False
                craving_inds = craving_inds[mask]
                cravings = cravings[craving_inds]
                self.craving_inds = craving_inds
                self.mean_craving[b, pid_num] = np.mean(cravings)
                self.std_craving[b, pid_num] = np.std(cravings)
                self.norm_cravings[b, pid_num, :] = stats.zscore(cravings)
                self.cravings[b, pid_num, :] = cravings

    @abstractmethod
    def update_Q(self, a, r, Qs, *args):
        pass

    def right_action_probs(self, actions, rewards, beta, cec_weight, *args):
        # Note that the first parameter is always the sample_beta, it is a required argument
        # Note that the second parameter is always the cec_weight, it is a required argument
        t_rewards = at.as_tensor_variable(rewards, dtype='int32')
        t_actions = at.as_tensor_variable(actions, dtype='int32')

        # Compute all loop vals
        # 0 - Q[left]
        # 1 - Q[right]
        # 2 - pred_craving
        # 3 - Q[t-1]
        # 4 - Q[t-2]
        # 5 - PE[t-1]
        # 6 - PE[t-2]
        loopvals =  at.zeros((7,), dtype='float64')
        loopvals, updates = aesara.scan(
            fn=self.update_Q,
            sequences=[t_actions, t_rewards],
            outputs_info=[loopvals],
            non_sequences=[*args])
        t_Qs = loopvals[:, :2]
        # t_pred_craving = pm.invlogit(loopvals[:, 2])
        t_pred_craving = pm.invlogit(loopvals[:, 2] + cec_weight*t_rewards)

        # Apply the sotfmax transformation
        t_Qs = t_Qs[:-1] * beta
        logp_actions = t_Qs - at.logsumexp(t_Qs, axis=1, keepdims=True)

        # Return the probabilities for the right action, in the original scale
        # Return predicted cravings
        return at.exp(logp_actions[:, 1]),  t_pred_craving
    
    def _load_act_rew_craving(self, pid_num, block, norm=True):
        pid = self.pid_list[pid_num]
        b = 0 if block=='money' else 1
        act = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].values
        rew = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].values
        if norm:
            crav = self.norm_cravings[b, pid_num, :]
        else:
            crav = self.cravings[b, pid_num, :]
        return act, rew, crav
    
    @abstractmethod
    def _define_priors(self):
        # Beta must be returned first!
        pass

    def fit(self, pid_num, block):
        pid = self.pid_list[pid_num]
        if self.save_path is not None:
            if not os.path.exists(f'{self.save_path}/{self.name}/'):
                os.makedirs(f'{self.save_path}/{self.name}/')
            filestr = f'{self.save_path}/{self.name}/{block}_{pid}.nc'
            if os.path.exists(filestr):
                print(f'PID: {pid}, Block {block} exists, loading from file...')
                self.traces[block][pid] = az.from_netcdf(filestr)
                return
        act, rew, cravings = self._load_act_rew_craving(pid_num, block, norm=False)
        with pm.Model() as model:
            priors = self._define_priors()
            action_probs, craving_pred = self.right_action_probs(act, rew, *priors)

            like = pm.Bernoulli('like', p=action_probs, observed=act[1:])
            probs_craving = pm.Deterministic('probs_craving', craving_pred[self.craving_inds-1])
            craving_like = pm.Binomial('craving_like', n=50, p=probs_craving, observed=cravings)
            
            self.traces[block][pid] = pm.sample(step=pm.Metropolis())
            # self.traces[block][pid].extend(pm.sample_prior_predictive())
            pm.sample_posterior_predictive(self.traces[block][pid], extend_inferencedata=True)
            if self.save_path is not None:
                self.traces[block][pid].to_netcdf(filestr)

## Inheritance models (EVRPE only)
### RW Models - Passive
class P_RW_0step(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_0step'
        self.type = 'passive'
        self.retro = 0
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'

    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * Qs[a] + w2 * pe)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_1stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_1stepMean'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - ONE STEP MEAN (Retro1) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3])/2 + w2 * (pe+Qs[5])/2)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_1stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_1stepDecay'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + w2*pe + (w2**2)*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_1stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_1stepSep'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, w3, w4 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*pe + w4*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, weight_three, weight_four

class P_RW_2stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_2stepMean'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3]+Qs[4])/3 + w2 * (pe+Qs[5]+Qs[6])/3)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_2stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_2stepDecay'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_2stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_2stepSep'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, w3, w4, w5, w6 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*Qs[4] + w4*pe + w5*Qs[5] + w6*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        weight_five = pm.Normal('weight_five', mu=0, sigma=1)
        weight_six = pm.Normal('weight_six', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, weight_three, weight_four, weight_five, weight_six

### RW Models - Active
class A_RW_0step(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RW_0step'
        self.type = 'active_lr'
        self.retro = 0
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'

    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + (alpha+bias) * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * Qs[a] + w2 * pe)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        # untr_mod = pm.Normal('untr_mod', mu=0, sigma=1)
        # mod = pm.Deterministic('mod', pm.math.invlogit(untr_mod))
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, mod

class A_RW_1stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RW_1stepMean'
        self.type = 'active_lr'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + (alpha+bias) * (pe))

        ## Calculate predicted craving - ONE STEP MEAN (Retro1) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3])/2 + w2 * (pe+Qs[5])/2)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, mod

class A_RW_1stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RW_1stepDecay'
        self.type = 'active_lr'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + (alpha+bias) * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + w2*pe + (w2**2)*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, mod

class A_RW_1stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RW_1stepSep'
        self.type = 'active_lr'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, w3, w4, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + (alpha+bias) * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*pe + w4*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, weight_three, weight_four, mod

class A_RW_2stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RW_2stepMean'
        self.type = 'active_lr'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + (alpha+bias) * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3]+Qs[4])/3 + w2 * (pe+Qs[5]+Qs[6])/3)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, mod

class A_RW_2stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RW_2stepDecay'
        self.type = 'active_lr'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + (alpha+bias) * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, mod

class A_RW_2stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RW_2stepSep'
        self.type = 'active_lr'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, w3, w4, w5, w6, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + (alpha+bias) * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*Qs[4] + w4*pe + w5*Qs[5] + w6*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        weight_five = pm.Normal('weight_five', mu=0, sigma=1)
        weight_six = pm.Normal('weight_six', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, weight_three, weight_four, weight_five, weight_six, mod

### RWSep Models - Passive
class P_RWSep_0step(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_0step'
        self.type = 'passive'
        self.retro = 0
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'

    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * Qs[a] + w2 * pe)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_1stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_1stepMean'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - ONE STEP MEAN (Retro1) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3])/2 + w2 * (pe+Qs[5])/2)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_1stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_1stepDecay'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + w2*pe + (w2**2)*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_1stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_1stepSep'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, w3, w4 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*pe + w4*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, weight_three, weight_four

class P_RWSep_2stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_2stepMean'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3]+Qs[4])/3 + w2 * (pe+Qs[5]+Qs[6])/3)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_2stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_2stepDecay'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_2stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_2stepSep'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, w3, w4, w5, w6 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*Qs[4] + w4*pe + w5*Qs[5] + w6*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        weight_five = pm.Normal('weight_five', mu=0, sigma=1)
        weight_six = pm.Normal('weight_six', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, weight_three, weight_four, weight_five, weight_six

### RWSep Models - Active
class A_RWSep_0step(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RWSep_0step'
        self.type = 'active_lr'
        self.retro = 0
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'

    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * Qs[a] + w2 * pe)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, mod

class A_RWSep_1stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RWSep_1stepMean'
        self.type = 'active_lr'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - ONE STEP MEAN (Retro1) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3])/2 + w2 * (pe+Qs[5])/2)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, mod

class A_RWSep_1stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RWSep_1stepDecay'
        self.type = 'active_lr'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + w2*pe + (w2**2)*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, mod

class A_RWSep_1stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RWSep_1stepSep'
        self.type = 'active_lr'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, w3, w4, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*pe + w4*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, weight_three, weight_four, mod

class A_RWSep_2stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RWSep_2stepMean'
        self.type = 'active_lr'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3]+Qs[4])/3 + w2 * (pe+Qs[5]+Qs[6])/3)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, mod

class A_RWSep_2stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RWSep_2stepDecay'
        self.type = 'active_lr'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, mod

class A_RWSep_2stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_RWSep_2stepSep'
        self.type = 'active_lr'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, w3, w4, w5, w6, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*Qs[4] + w4*pe + w5*Qs[5] + w6*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        weight_five = pm.Normal('weight_five', mu=0, sigma=1)
        weight_six = pm.Normal('weight_six', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, weight_three, weight_four, weight_five, weight_six, mod

## Mixed EVRPE-CEC prototype class, designed for active beta models
class MixedPrototype_Beta(ABC):
    def __init__(self, longform, summary, project_dir, save_path):
        self.name = None        
        self.longform = longform
        self.summary = summary
        self.pid_list = longform['PID'].unique()
        self.traces = {
            'money': {},
            'other': {}
        }
        self.project_dir = project_dir

        num_craving_trials = 20
        num_blocks = 2
        self.craving_inds = None
        self.mean_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.std_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.norm_cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self.cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self._calc_norm_cravings()

        self.save_path = save_path
    
    def _calc_norm_cravings(self):
        for pid_num in range(len(self.pid_list)):
            for b, block in enumerate(['money', 'other']):
                pid = self.pid_list[pid_num]
                cravings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values
                craving_inds = np.squeeze(np.argwhere(cravings>-1))
                mask = np.ones(len(craving_inds), dtype=bool)
                mask[12] = False
                craving_inds = craving_inds[mask]
                cravings = cravings[craving_inds]
                self.craving_inds = craving_inds
                self.mean_craving[b, pid_num] = np.mean(cravings)
                self.std_craving[b, pid_num] = np.std(cravings)
                self.norm_cravings[b, pid_num, :] = stats.zscore(cravings)
                self.cravings[b, pid_num, :] = cravings

    @abstractmethod
    def update_Q(self, a, r, Qs, *args):
        pass

    def right_action_probs(self, actions, rewards, beta, mod, cec_weight, *args):
        # Note that the first parameter is always the sample_beta, it is a required argument
        # Note that the second parameter is always the cec_weight, it is a required argument
        t_rewards = at.as_tensor_variable(rewards, dtype='int32')
        t_actions = at.as_tensor_variable(actions, dtype='int32')

        # Compute all loop vals
        # 0 - Q[left]
        # 1 - Q[right]
        # 2 - pred_craving
        # 3 - Q[t-1]
        # 4 - Q[t-2]
        # 5 - PE[t-1]
        # 6 - PE[t-2]
        loopvals =  at.zeros((7,), dtype='float64')
        loopvals, updates = aesara.scan(
            fn=self.update_Q,
            sequences=[t_actions, t_rewards],
            outputs_info=[loopvals],
            non_sequences=[*args])
        t_Qs = loopvals[:, :2]
        # t_pred_craving = pm.invlogit(loopvals[:, 2])
        t_pred_craving = pm.invlogit(loopvals[:, 2] + cec_weight*t_rewards)

        ## Compute the beta bias term
        biased_beta = mod*pm.math.invlogit(t_Qs[2]) + beta

        # Apply the sotfmax transformation
        t_Qs = t_Qs[:-1] * biased_beta[:-1]
        logp_actions = t_Qs - at.logsumexp(t_Qs, axis=1, keepdims=True)

        # Return the probabilities for the right action, in the original scale
        # Return predicted cravings
        return at.exp(logp_actions[:, 1]),  t_pred_craving
    
    def _load_act_rew_craving(self, pid_num, block, norm=True):
        pid = self.pid_list[pid_num]
        b = 0 if block=='money' else 1
        act = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].values
        rew = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].values
        if norm:
            crav = self.norm_cravings[b, pid_num, :]
        else:
            crav = self.cravings[b, pid_num, :]
        return act, rew, crav
    
    @abstractmethod
    def _define_priors(self):
        # Beta must be returned first!
        pass

    def fit(self, pid_num, block):
        pid = self.pid_list[pid_num]
        if self.save_path is not None:
            if not os.path.exists(f'{self.save_path}/{self.name}/'):
                os.makedirs(f'{self.save_path}/{self.name}/')
            filestr = f'{self.save_path}/{self.name}/{block}_{pid}.nc'
            if os.path.exists(filestr):
                print(f'PID: {pid}, Block {block} exists, loading from file...')
                self.traces[block][pid] = az.from_netcdf(filestr)
                return
        act, rew, cravings = self._load_act_rew_craving(pid_num, block, norm=False)
        with pm.Model() as model:
            priors = self._define_priors()
            action_probs, craving_pred = self.right_action_probs(act, rew, *priors)

            like = pm.Bernoulli('like', p=action_probs, observed=act[1:])
            probs_craving = pm.Deterministic('probs_craving', craving_pred[self.craving_inds-1])
            craving_like = pm.Binomial('craving_like', n=50, p=probs_craving, observed=cravings)
            
            self.traces[block][pid] = pm.sample(step=pm.Metropolis())
            # self.traces[block][pid].extend(pm.sample_prior_predictive())
            pm.sample_posterior_predictive(self.traces[block][pid], extend_inferencedata=True)
            if self.save_path is not None:
                self.traces[block][pid].to_netcdf(filestr)

### RW Models - Active Beta
class A_Beta_RW_0step(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RW_0step'
        self.type = 'active_beta'
        self.retro = 0
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'

    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha*pe)

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * Qs[a] + w2 * pe)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        # untr_mod = pm.Normal('untr_mod', mu=0, sigma=1)
        # mod = pm.Deterministic('mod', pm.math.invlogit(untr_mod))
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha, weight_zero, weight_one, weight_two

class A_Beta_RW_1stepMean(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RW_1stepMean'
        self.type = 'active_beta'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha*pe)

        ## Calculate predicted craving - ONE STEP MEAN (Retro1) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3])/2 + w2 * (pe+Qs[5])/2)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha, weight_zero, weight_one, weight_two

class A_Beta_RW_1stepDecay(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RW_1stepDecay'
        self.type = 'active_beta'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha*pe)

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + w2*pe + (w2**2)*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha, weight_zero, weight_one, weight_two

class A_Beta_RW_1stepSep(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RW_1stepSep'
        self.type = 'active_beta'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, w3, w4 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha*pe)

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*pe + w4*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha, weight_zero, weight_one, weight_two, weight_three, weight_four

class A_Beta_RW_2stepMean(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RW_2stepMean'
        self.type = 'active_beta'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha*pe)

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3]+Qs[4])/3 + w2 * (pe+Qs[5]+Qs[6])/3)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha, weight_zero, weight_one, weight_two

class A_Beta_RW_2stepDecay(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RW_2stepDecay'
        self.type = 'active_beta'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha*pe)

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha, weight_zero, weight_one, weight_two

class A_Beta_RW_2stepSep(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RW_2stepSep'
        self.type = 'active_beta'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, w3, w4, w5, w6 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha*pe)

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*Qs[4] + w4*pe + w5*Qs[5] + w6*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        weight_five = pm.Normal('weight_five', mu=0, sigma=1)
        weight_six = pm.Normal('weight_six', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha, weight_zero, weight_one, weight_two, weight_three, weight_four, weight_five, weight_six

### RWSep Models - Active
class A_Beta_RWSep_0step(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RWSep_0step'
        self.type = 'active_beta'
        self.retro = 0
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'

    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * Qs[a] + w2 * pe)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class A_Beta_RWSep_1stepMean(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RWSep_1stepMean'
        self.type = 'active_beta'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - ONE STEP MEAN (Retro1) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3])/2 + w2 * (pe+Qs[5])/2)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class A_Beta_RWSep_1stepDecay(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RWSep_1stepDecay'
        self.type = 'active_beta'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + w2*pe + (w2**2)*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class A_Beta_RWSep_1stepSep(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RWSep_1stepSep'
        self.type = 'active_beta'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, w3, w4 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*pe + w4*Qs[5]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, weight_three, weight_four

class A_Beta_RWSep_2stepMean(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RWSep_2stepMean'
        self.type = 'active_beta'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3]+Qs[4])/3 + w2 * (pe+Qs[5]+Qs[6])/3)

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class A_Beta_RWSep_2stepDecay(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RWSep_2stepDecay'
        self.type = 'active_beta'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class A_Beta_RWSep_2stepSep(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'A_Beta_RWSep_2stepSep'
        self.type = 'active_beta'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, w3, w4, w5, w6 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*Qs[4] + w4*pe + w5*Qs[5] + w6*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        weight_five = pm.Normal('weight_five', mu=0, sigma=1)
        weight_six = pm.Normal('weight_six', mu=0, sigma=1)
        mod = pm.LogNormal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, weight_three, weight_four, weight_five, weight_six

## NULL MODELS
class BiasedCEC:
    def __init__(self, longform, summary, project_dir, save_path):
        self.name = 'Biased_CEC'        
        self.longform = longform
        self.summary = summary
        self.pid_list = longform['PID'].unique()
        self.traces = {
            'money': {},
            'other': {}
        }
        self.project_dir = project_dir

        self.type = 'null'
        self.retro = 0
        self.decision = 'Biased'
        self.craving = 'CEC'

        num_craving_trials = 20
        num_blocks = 2
        self.craving_inds = None
        self.mean_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.std_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.norm_cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self.cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self._calc_norm_cravings()

        self.save_path = save_path
    
    def _calc_norm_cravings(self):
        for pid_num in range(len(self.pid_list)):
            for b, block in enumerate(['money', 'other']):
                pid = self.pid_list[pid_num]
                cravings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values
                craving_inds = np.squeeze(np.argwhere(cravings>-1))
                mask = np.ones(len(craving_inds), dtype=bool)
                mask[12] = False
                craving_inds = craving_inds[mask]
                cravings = cravings[craving_inds]
                self.craving_inds = craving_inds
                self.mean_craving[b, pid_num] = np.mean(cravings)
                self.std_craving[b, pid_num] = np.std(cravings)
                self.norm_cravings[b, pid_num, :] = stats.zscore(cravings)
                self.cravings[b, pid_num, :] = cravings

    def right_action_probs(self, actions, rewards, *args):
        
        sample_bias, weight_zero, weight_one = args

        t_rewards = at.as_tensor_variable(rewards, dtype='int32')
        t_actions = at.as_tensor_variable(actions, dtype='int32')

        probs = at.repeat(sample_bias, t_actions.shape[0])
        pred_cravings = weight_zero + weight_one*t_rewards

        return probs[1:],  pm.invlogit(pred_cravings)
    
    def _load_act_rew_craving(self, pid_num, block, norm=True):
        pid = self.pid_list[pid_num]
        b = 0 if block=='money' else 1
        act = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].values
        rew = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].values
        if norm:
            crav = self.norm_cravings[b, pid_num, :]
        else:
            crav = self.cravings[b, pid_num, :]
        return act, rew, crav
    
    def _define_priors(self):
        untr_bias = pm.Normal('untr_bias', mu=0, sigma=1)
        bias = pm.Deterministic('bias', pm.math.invlogit(untr_bias))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        return bias, weight_zero, weight_one
        

    def fit(self, pid_num, block):
        pid = self.pid_list[pid_num]
        if self.save_path is not None:
            if not os.path.exists(f'{self.save_path}/{self.name}/'):
                os.makedirs(f'{self.save_path}/{self.name}/')
            filestr = f'{self.save_path}/{self.name}/{block}_{pid}.nc'
            if os.path.exists(filestr):
                print(f'PID: {pid}, Block {block} exists, loading from file...')
                self.traces[block][pid] = az.from_netcdf(filestr)
                return
        act, rew, cravings = self._load_act_rew_craving(pid_num, block, norm=False)
        with pm.Model() as model:
            priors = self._define_priors()
            action_probs, craving_pred = self.right_action_probs(act, rew, *priors)

            like = pm.Bernoulli('like', p=action_probs, observed=act[1:])
            probs_craving = pm.Deterministic('probs_craving', craving_pred[self.craving_inds-1])
            craving_like = pm.Binomial('craving_like', n=50, p=probs_craving, observed=cravings)
            
            self.traces[block][pid] = pm.sample(step=pm.Metropolis())
            # self.traces[block][pid].extend(pm.sample_prior_predictive())
            pm.sample_posterior_predictive(self.traces[block][pid], extend_inferencedata=True)
            if self.save_path is not None:
                self.traces[block][pid].to_netcdf(filestr)

class HeuCEC:
    def __init__(self, longform, summary, project_dir, save_path):
        self.name = 'Heuristic_CEC'        
        self.longform = longform
        self.summary = summary
        self.pid_list = longform['PID'].unique()
        self.traces = {
            'money': {},
            'other': {}
        }
        self.project_dir = project_dir

        self.type = 'null'
        self.retro = 0
        self.decision = 'Heuristic'
        self.craving = 'CEC'

        num_craving_trials = 20
        num_blocks = 2
        self.craving_inds = None
        self.mean_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.std_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.norm_cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self.cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self._calc_norm_cravings()

        self.save_path = save_path
    
    def _calc_norm_cravings(self):
        for pid_num in range(len(self.pid_list)):
            for b, block in enumerate(['money', 'other']):
                pid = self.pid_list[pid_num]
                cravings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values
                craving_inds = np.squeeze(np.argwhere(cravings>-1))
                mask = np.ones(len(craving_inds), dtype=bool)
                mask[12] = False
                craving_inds = craving_inds[mask]
                cravings = cravings[craving_inds]
                self.craving_inds = craving_inds
                self.mean_craving[b, pid_num] = np.mean(cravings)
                self.std_craving[b, pid_num] = np.std(cravings)
                self.norm_cravings[b, pid_num, :] = stats.zscore(cravings)
                self.cravings[b, pid_num, :] = cravings

    def add_eps(self, st, a, eps_t, eps):
        return aesara.ifelse.ifelse(
                at.eq(st, 1),
                aesara.ifelse.ifelse(
                    at.eq(a, 1),
                    eps,
                    1-eps
                ),
                aesara.ifelse.ifelse(
                    at.eq(a, 1),
                    1-eps,
                    eps
                )
            )
    
    def right_action_probs(self, actions, rewards, strat, *args):
        
        sample_eps, weight_zero, weight_one = args

        t_strat = at.as_tensor_variable(strat, dtype='int32')
        t_actions = at.as_tensor_variable(actions, dtype='int32')
        t_rewards = at.as_tensor_variable(rewards, dtype='int32')

        # Compute the Qs values
        t_eps = at.as_tensor_variable(np.asarray(1, 'float64'))
        t_eps, updates = aesara.scan(
            fn=self.add_eps,
            sequences=[t_strat, t_actions],
            outputs_info=t_eps,
            non_sequences=[sample_eps])
        
        pred_cravings = weight_zero + weight_one*t_rewards

        return t_eps[1:],  pm.invlogit(pred_cravings)
    
    def _load_act_rew_craving(self, pid_num, block, norm=True):
        pid = self.pid_list[pid_num]
        b = 0 if block=='money' else 1
        act = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].values
        rew = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].values
        if norm:
            crav = self.norm_cravings[b, pid_num, :]
        else:
            crav = self.cravings[b, pid_num, :]
        return act, rew, crav
    
    def _define_priors(self):
        untr_eps = pm.Normal('untr_eps', mu=0, sigma=1)
        eps = pm.Deterministic('eps', pm.math.invlogit(untr_eps))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        return eps, weight_zero, weight_one
        

    def fit(self, pid_num, block):
        pid = self.pid_list[pid_num]
        if self.save_path is not None:
            if not os.path.exists(f'{self.save_path}/{self.name}/'):
                os.makedirs(f'{self.save_path}/{self.name}/')
            filestr = f'{self.save_path}/{self.name}/{block}_{pid}.nc'
            if os.path.exists(filestr):
                print(f'PID: {pid}, Block {block} exists, loading from file...')
                self.traces[block][pid] = az.from_netcdf(filestr)
                return
        act, rew, cravings = self._load_act_rew_craving(pid_num, block, norm=False)
        strat = np.zeros(len(act))
        for i, a in enumerate(act):
            if i < 2:
                continue
            should_switch = np.all(np.array([rew[i-2]==rew[i-1], rew[i-1]==0]), axis=0)
            do_switch = act[i-1]!=act[i]
            strat[i] = should_switch==do_switch
        strat = strat.astype(int)
        with pm.Model() as model:
            priors = self._define_priors()
            action_probs, craving_pred = self.right_action_probs(act, rew, strat, *priors)

            like = pm.Bernoulli('like', p=action_probs, observed=act[1:])
            probs_craving = pm.Deterministic('probs_craving', craving_pred[self.craving_inds-1])
            craving_like = pm.Binomial('craving_like', n=50, p=probs_craving, observed=cravings)
            
            self.traces[block][pid] = pm.sample(step=pm.Metropolis())
            # self.traces[block][pid].extend(pm.sample_prior_predictive())
            pm.sample_posterior_predictive(self.traces[block][pid], extend_inferencedata=True)
            if self.save_path is not None:
                self.traces[block][pid].to_netcdf(filestr)

class RWCEC():
    def __init__(self, longform, summary, project_dir, save_path):
        self.name = 'RW_CEC'        
        self.longform = longform
        self.summary = summary
        self.pid_list = longform['PID'].unique()
        self.traces = {
            'money': {},
            'other': {}
        }
        self.project_dir = project_dir

        self.type = 'null'
        self.retro = 0
        self.decision = 'RW'
        self.craving = 'CEC'

        num_craving_trials = 20
        num_blocks = 2
        self.craving_inds = None
        self.mean_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.std_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.norm_cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self.cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self._calc_norm_cravings()

        self.save_path = save_path
    
    def _calc_norm_cravings(self):
        for pid_num in range(len(self.pid_list)):
            for b, block in enumerate(['money', 'other']):
                pid = self.pid_list[pid_num]
                cravings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values
                craving_inds = np.squeeze(np.argwhere(cravings>-1))
                mask = np.ones(len(craving_inds), dtype=bool)
                mask[12] = False
                craving_inds = craving_inds[mask]
                cravings = cravings[craving_inds]
                self.craving_inds = craving_inds
                self.mean_craving[b, pid_num] = np.mean(cravings)
                self.std_craving[b, pid_num] = np.std(cravings)
                self.norm_cravings[b, pid_num, :] = stats.zscore(cravings)
                self.cravings[b, pid_num, :] = cravings

    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))
        return Qs

    def right_action_probs(self, actions, rewards, beta, *args):
        alpha, weight_zero, weight_one = args
        t_rewards = at.as_tensor_variable(rewards, dtype='int32')
        t_actions = at.as_tensor_variable(actions, dtype='int32')

        # Compute all loop vals
        loopvals =  at.zeros((7,), dtype='float64')
        loopvals, updates = aesara.scan(
            fn=self.update_Q,
            sequences=[t_actions, t_rewards],
            outputs_info=[loopvals],
            non_sequences=[*args])
        t_Qs = loopvals[:, :2]
        # t_pred_craving = pm.invlogit(loopvals[:, 2])
        t_pred_craving = weight_zero + weight_one*t_rewards

        # Apply the sotfmax transformation
        t_Qs = t_Qs[:-1] * beta
        logp_actions = t_Qs - at.logsumexp(t_Qs, axis=1, keepdims=True)

        # Return the probabilities for the right action, in the original scale
        # Return predicted cravings
        return at.exp(logp_actions[:, 1]),  pm.math.invlogit(t_pred_craving)
    
    def _load_act_rew_craving(self, pid_num, block, norm=True):
        pid = self.pid_list[pid_num]
        b = 0 if block=='money' else 1
        act = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].values
        rew = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].values
        if norm:
            crav = self.norm_cravings[b, pid_num, :]
        else:
            crav = self.cravings[b, pid_num, :]
        return act, rew, crav
    
    @abstractmethod
    def _define_priors(self):
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        return beta, alpha, weight_zero, weight_one

    def fit(self, pid_num, block):
        pid = self.pid_list[pid_num]
        if self.save_path is not None:
            if not os.path.exists(f'{self.save_path}/{self.name}/'):
                os.makedirs(f'{self.save_path}/{self.name}/')
            filestr = f'{self.save_path}/{self.name}/{block}_{pid}.nc'
            if os.path.exists(filestr):
                print(f'PID: {pid}, Block {block} exists, loading from file...')
                self.traces[block][pid] = az.from_netcdf(filestr)
                return
        act, rew, cravings = self._load_act_rew_craving(pid_num, block, norm=False)
        with pm.Model() as model:
            priors = self._define_priors()
            action_probs, craving_pred = self.right_action_probs(act, rew, *priors)

            like = pm.Bernoulli('like', p=action_probs, observed=act[1:])
            probs_craving = pm.Deterministic('probs_craving', craving_pred[self.craving_inds-1])
            craving_like = pm.Binomial('craving_like', n=50, p=probs_craving, observed=cravings)
            
            self.traces[block][pid] = pm.sample(step=pm.Metropolis())
            # self.traces[block][pid].extend(pm.sample_prior_predictive())
            pm.sample_posterior_predictive(self.traces[block][pid], extend_inferencedata=True)
            if self.save_path is not None:
                self.traces[block][pid].to_netcdf(filestr)

## Batchfit class
class BatchFit(object):
    def __init__(self, model_list, longform, df_summary, project_dir, save_path):
        self.models = {}
        for model_name in model_list:
            self.models[model_name] = eval(f'{model_name}(longform, df_summary, project_dir, save_path)')
        self.longform = longform
        self.df_summary = df_summary
        self.project_dir = project_dir
    
    def fit(self, pid_num, block, jupyter=False):
        for model_name in self.models:
            if jupyter:
                clear_output(wait=True)
            print(f'Fitting {model_name}: PID - {pid_num}, Block - {block}')
            self.models[model_name].fit(pid_num, block)
    
    def join(self, batch):
        for model_name in batch.models:
            if model_name not in self.models:
                self.models[model_name] = batch.models[model_name]
            else:
                print(f'Model {model_name} already exists in this batch')
                
