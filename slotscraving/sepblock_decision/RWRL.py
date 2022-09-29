## BAYESIAN LIBRARIES
from multiprocessing.sharedctypes import Value
import pymc3 as pm
import theano.tensor as tt
import theano
import arviz as az

## STANDARD LIBRARIES
import numpy as np
import pandas as pd
from scipy.special import logsumexp

## UTILITIES
from IPython.display import clear_output
import os

from .Prototype import Prototype

# DETERMINE LIKELIHOOD OF PARAMETER
def update_func(action, reward, Qs, alpha_pos, alpha_neg):
    """This function updates the Q table according to the RL update rule. 
    It will be called by theano.scan to do so recursively, given the observed data and the alpha parameter
    THIS LIKELY NEEDS TO BE UDPATED FOR EVERY MODEL.

    Args:
        action (theano.tensor.imatrix): Theano matrix of actions for each block
        reward (theano.tensor.imatrix): Theano matrix of rewards for each block
        Qs (theano.tensor.fmatrix): Theano matrix of values for each block and choice
        alpha (theano.tensor.fvector): Theano vector of alphas for each block

    Returns:
        theano.tensor.fmatrix: Updated matrix of values
    """
    # PE = reward - Qs[action]
    # Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * PE)
    # return Qs

    PE = reward - Qs[action]
    trial_alpha = tt.stack(alpha_pos, alpha_neg)[tt.cast(PE<0, 'int32')]
    Qs = tt.set_subtensor(Qs[action], Qs[action] + trial_alpha * PE)
    return Qs

class RWRL(Prototype):
    def __init__(self, model_name, save_path, load=False, summary=None, longform=None):
        """Model class. Accepts the summary and longform dataframes as inputs. 
        Can also load from netcdf folder, as long as param, summary, longform csvs, and 
        netcdf model traces are available. Likely DOES NOT NEED TO GET MODIFIED beyond model name.

        Args:
            summary (pd.DataFrame, optional): Summary dataframe for subjects. Defaults to None.
            longform (pd.DataFrame, optional): Longform data for all subjects. Defaults to None.
            load_from (str, optional): Path to saved netcdf folder. Defaults to None.
            model_name (str, optional): Description of model. Defaults to None.

        Raises:
            ValueError: Raise error if model name is not given.
        """

        self.model_description = 'RW model with separate learning rates for +/- PEs'
        self.params = ['alpha_pos', 'alpha_neg', 'beta']
        super().__init__(
            model_name=model_name,
            save_path=save_path,
            load=load,
            summary=summary,
            longform=longform,
        )


    ## Theano-PyMC wrapper to calculate Qs for actions and rewards
    def _theano_llik(self, _alpha_pos, _alpha_neg, _beta, _actions, _rewards):
        """Wrapper for calculating Qs for Rescorla-Wagner variants.
        This function likely DOES NOT need to get modified.

        Args:
            _alpha (theano.tensor.fvector): Theano vector of alphas for each block
            _beta (theano.tensor.fvector): Theano vector of betas for each block
            _actions (theano.tensor.imatrix): Theano matrix of actions for each block
            _rewards (theano.tensor.imatrix): Theano matrix of rewards for each block

        Returns:
            theano.tensor.iscalar: Log-likelihood of chosen set of alphas and betas
        """
        rw = tt.cast(_rewards, "int16")
        act = tt.cast(_actions, "int16")
        beta_stack = tt.repeat(_beta, self.n_trials * self.n_options).reshape(
            (self.n_options, self.n_trials)
        )

        # Compute the Qs values
        Qs = 0.5 * tt.ones(2, dtype="float64")

        Qs, updates = theano.scan(
            fn=update_func,
            sequences=[act, rw],
            outputs_info=[Qs],
            non_sequences=[_alpha_pos, _alpha_neg],
        )
        Qs = tt.transpose(Qs)

        # Apply the sotfmax transformation
        Qs_ = tt.mul(beta_stack, Qs)
        log_prob_actions = Qs_ - pm.math.logsumexp(Qs_)
        return tt.sum(
            log_prob_actions[act, tt.arange(self.n_trials)]
        )  # PyMC makes it negative by default

    def _fit(
        self,
        pid,
        draws=1000,
        chains=2,
        cores=4,
        rerun=False,
        jupyter=False,
    ):

        # fmt: off
        if pid not in self.traces.keys():
            self.traces[pid] = {}

        for block in ['money', 'other']:
            if os.path.exists(f'{self.save_path}/{self.model_name}/traces/{pid}_{block}.nc') and not rerun:
                if jupyter:
                    print(f'Participant {pid}, {block} block completed...')
                if block not in self.traces[pid]:
                    self.traces[pid][block] = az.from_netcdf(f'{self.save_path}/{self.model_name}/traces/{pid}_{block}.nc')
                continue
            elif (pid, block) in self.skip_list:
                if jupyter:
                    print(f'Participant {pid}, {block} block skipped...')
                continue
            
            try:
                actions = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].to_numpy().astype(int)
                rewards = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].to_numpy().astype(int)
                with pm.Model() as m:
                    alpha_pos = pm.Normal(f"alpha_pos", 0.5, 0.25)
                    alpha_neg = pm.Normal(f"alpha_neg", 0.5, 0.25)
                    beta = pm.Gamma(f"beta", alpha=3, beta=1.5)
                    like = pm.DensityDist('like', self._theano_llik, observed=dict(_alpha_pos=alpha_pos, _alpha_neg=alpha_neg, _beta=beta, _actions=actions, _rewards=rewards))
                    trace = pm.sample(start={'alpha_pos': 0.5, 'alpha_neg': 0.5, 'beta': 3},
                        draws=draws, chains=chains, cores=cores, 
                        return_inferencedata=True, idata_kwargs={"density_dist_obs": False}
                    )
                trace.to_netcdf(f'{self.save_path}/{self.model_name}/traces/{pid}_{block}.nc')
                print(f'Participant {pid}, {block} block completed...')
            except RuntimeError as e:
                print(e)
                print(f'Participant {pid}, {block} block failed...')
                self.skip_list.append((pid, block))
                continue
        # fmt: on


    def calc_Q_table(self):
        """This function calculates mean alphas across traces for each parameter, and recalculates and stores Qs
        based on mean estimated parameter values. THIS FUNCTION LIKELY NEEDS TO BE UPDATED FOR EACH MODEL.
        """
        Q_table = pd.DataFrame()
        for pid in self.summary['PID']:

            # fmt: off
            ## CALCULATE MEANS OF PARAMS: NEED TO CHANGE
            mean_alphas_pos = np.vstack([
                self.traces[pid]['money'].posterior.alpha_pos.to_numpy().mean(axis=1).mean(axis=0),
                self.traces[pid]['other'].posterior.alpha_pos.to_numpy().mean(axis=1).mean(axis=0)
            ])
            mean_alphas_neg = mean_alphas_pos = np.vstack([
                self.traces[pid]['money'].posterior.alpha_neg.to_numpy().mean(axis=1).mean(axis=0),
                self.traces[pid]['other'].posterior.alpha_neg.to_numpy().mean(axis=1).mean(axis=0)
            ])
            # mean_betas = model.traces[pid].posterior.beta.to_numpy().mean(axis=1).mean(axis=0)
            
            actions = np.vstack([
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Action'].to_numpy().astype(int),
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Action'].to_numpy().astype(int)
            ])
            rewards = np.vstack([
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Reward'].to_numpy().astype(int),
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Reward'].to_numpy().astype(int)
            ])
            
            ## UPDATE VALUES: NEED TO UPDATE
            Qs = np.ones(shape=(actions.shape[0], actions.shape[1], 2)) * 0.5
            PEs = np.zeros(shape=(actions.shape[0], actions.shape[1]))
            for b, block in enumerate(['money', 'other']):
                block_actions = actions[b]
                block_rewards = rewards[b]

                for t, (act, rw) in enumerate(zip(block_actions, block_rewards)):
                    PEs[b, t] = rw - Qs[b,t-1,act]
                    if PEs[b, t] > 0:
                        Qs[b,t,act] = Qs[b,t-1,act] + mean_alphas_pos[b] * PEs[b, t]
                    else:
                        Qs[b,t,act] = Qs[b,t-1,act] + mean_alphas_neg[b] * PEs[b, t]
                    Qs[b,t,1-act] = Qs[b,t-1,1-act]

            participant_Q_table = pd.DataFrame.from_dict({
                'PID': [pid]*actions.shape[1]*2,
                'Trial': np.hstack([np.arange(actions.shape[1])+1,np.arange(actions.shape[1])+1]),
                'Q_left': np.hstack([Qs[0,:,0], Qs[1,:,0]]),
                'Q_right': np.hstack([Qs[0,:,1], Qs[1,:,1]]),
                'PE': np.hstack([PEs[0, :], PEs[1, :]]),
                'Type': np.hstack([np.array(['money']*60), np.array(['other']*60)])
            })
            Q_table = pd.concat([Q_table, participant_Q_table])
        
        self.longform = self.longform.merge(Q_table)
        self.longform.to_csv(f'{self.save_path}/{self.model_name}/longform.csv', index=False)
        # fmt: on

    def _calc_lik(self, pid, block):
        alpha_pos_values = self.traces[pid][block].posterior.alpha_pos.values.flatten()
        alpha_neg_values = self.traces[pid][block].posterior.alpha_neg.values.flatten()
        beta_values = self.traces[pid][block].posterior.beta.values.flatten()
        actions = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].to_numpy().astype(int)
        rewards = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].to_numpy().astype(int)
        lik = []
        for alpha_pos, alpha_neg, beta in zip(alpha_pos_values, alpha_neg_values, beta_values):
            Qs = np.ones((len(actions) + 1, 2), dtype=float) * 0.5
            for t,(a,r) in enumerate(zip(actions, rewards)):
                delta = r - Qs[t, a]
                if delta > 0:
                    Qs[t + 1, a] = Qs[t, a] + alpha_pos * delta
                else:
                    Qs[t + 1, a] = Qs[t, a] + alpha_neg * delta
                Qs[t + 1, 1 - a] = Qs[t, 1 - a]
            # Apply the softmax transformation in a vectorized way to the values
            Qs_ = Qs * beta
            log_prob_actions = Qs_ - logsumexp(Qs_, axis=1)[:, None]
            log_prob_actions = log_prob_actions[np.arange(len(actions)), actions]
            lik.append(np.sum(log_prob_actions[1:]))
        return np.array(lik)

    # def calc_bics(self):
    #     self.bics = {}
    #     for pid in self.summary['PID']:
    #         self.bics[pid] = []
    #         actions = np.vstack([
    #             self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Action'].to_numpy().astype(int),
    #             self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Action'].to_numpy().astype(int),
    #             # self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='numberbar_mixed')]['Action'].to_numpy().astype(int)
    #         ])
    #         rewards = np.vstack([
    #             self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Reward'].to_numpy().astype(int),
    #             self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Reward'].to_numpy().astype(int),
    #             # self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='numberbar_mixed')]['Reward'].to_numpy().astype(int)
    #         ])
    #         for alpha_pos_chain, alpha_neg_chain, beta_chain in zip(
    #             self.traces[pid].posterior.alpha_pos.values, 
    #             self.traces[pid].posterior.alpha_neg.values,
    #             self.traces[pid].posterior.beta.values):
    #             for alpha_pos_block_params, alpha_neg_block_params, beta_block_params in zip(alpha_pos_chain, alpha_neg_chain, beta_chain):
    #                 sample_bic = []
    #                 for b, block in enumerate(['money', 'other']):
    #                     alpha_pos_block_param = alpha_pos_block_params[b]
    #                     alpha_neg_block_param = alpha_neg_block_params[b]
    #                     beta_block_param = beta_block_params[b]
    #                     block_actions = actions[b]
    #                     block_rewards = rewards[b]
                        
    #                     Qs = np.ones((len(block_actions) + 1, 2), dtype=float) * 0.5
    #                     for t, (a, r) in enumerate(zip(block_actions, block_rewards)):
    #                         delta = r - Qs[t, a]
    #                         if delta >= 0:
    #                             Qs[t + 1, a] = Qs[t, a] + alpha_pos_block_param * delta
    #                         elif delta < 0:
    #                             Qs[t + 1, a] = Qs[t, a] + alpha_neg_block_param * delta
    #                         Qs[t + 1, 1 - a] = Qs[t, 1 - a]

    #                     # Apply the softmax transformation in a vectorized way to the values
    #                     Qs_ = Qs * beta_block_param
    #                     log_prob_actions = Qs_ - logsumexp(Qs_, axis=1)[:, None]
    #                     log_prob_actions = log_prob_actions[np.arange(len(block_actions)), block_actions]

    #                     lik = -np.sum(log_prob_actions[1:])
    #                     bic = np.log(len(block_actions)) * self.n_params + 2 * lik
    #                     sample_bic.append(bic)
    #                 self.bics[pid].append(sample_bic)
    #         self.bics[pid] = np.array(self.bics[pid])

    # def save_to_netcdf(self, path, model_name):
    #     """Save the parameters and traces of the model to a path.

    #     Args:
    #         path (str): Path to Netcdf folder
    #         model_name (str): Name of the model

    #     Raises:
    #         ValueError: The traces need to be calculated before saving.
    #     """
    #     if not self.fit_complete:
    #         raise ValueError('Model has not been fit to data')

    #     if not os.path.exists(f'{path}/{model_name}/'):
    #         os.makedirs(f'{path}/{model_name}/')
    #     if not os.path.exists(f'{path}/{model_name}/traces/'):
    #         os.makedirs(f'{path}/{model_name}/traces/')

    #     pd.DataFrame({
    #         'model_description': self.model_description,
    #         'n_subj': self.n_subj,
    #         'n_trials': self.n_trials,
    #         'n_blocks': self.n_blocks,
    #         'n_options': self.n_options,
    #         'fit_complete': self.fit_complete,
    #     }, index=[0]).to_csv(f'{path}/{model_name}/params.csv', index=False)

    #     self.summary.to_csv(f'{path}/{model_name}/summary.csv', index=False)
    #     self.longform.to_csv(f'{path}/{model_name}/longform.csv', index=False)
    #     # self.Q_table.to_csv(f'{path}/{model_name}/derivatives/Q_table.csv', index=False)

    #     for pid in self.summary['PID']:
    #         self.traces[pid].to_netcdf(f'{path}/{model_name}/traces/{pid}.nc')

    # def divergences(self):
    #     divergences = []
    #     for pid in self.summary['PID']:
    #         div = self.traces[pid].sample_stats.diverging.to_numpy().sum()
    #         if div > 40:
    #             divergences.append([pid, div, 'Major Divergences'])
    #         elif div > 10:
    #             divergences.append([pid, div, 'Minor'])
    #     divergences = np.array(divergences)
    #     if divergences.size>0:
    #         return pd.DataFrame.from_dict({
    #             'PID': divergences[:,0],
    #             'Divergences': divergences[:,1],
    #             'Level': divergences[:,2]
    #         })
    #     else:
    #         return 'No divergences'