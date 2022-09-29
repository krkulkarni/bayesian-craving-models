## BAYESIAN LIBRARIES
import pymc3 as pm
import theano.tensor as tt
import theano
import arviz as az

## STANDARD LIBRARIES
import numpy as np
import pandas as pd
from scipy.special import logsumexp

## UTILITIES
from IPython.display import clear_output, display
import os

from .Prototype import Prototype

# DETERMINE LIKELIHOOD OF PARAMETER
def update_func(action, reward, Qs, alpha):
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
    PE = reward - Qs[action]
    Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * PE)
    return Qs


class RW(Prototype):
    def __init__(self, model_name, save_path, load=False, summary=None, longform=None):
        """RW model class. Accepts the summary and longform dataframes as inputs. 
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

        self.model_description = "Standard Rescorla-Wagner"
        self.params = ["alpha", "beta"]
        super().__init__(
            model_name=model_name,
            save_path=save_path,
            load=load,
            summary=summary,
            longform=longform,
        )

    ## Theano-PyMC wrapper to calculate Qs for actions and rewards
    def _theano_llik(self, _alpha, _beta, _actions, _rewards):
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
            non_sequences=[_alpha],
        )
        Qs = tt.transpose(Qs)

        # Apply the sotfmax transformation
        Qs_ = tt.mul(beta_stack, Qs)
        log_prob_actions = Qs_ - pm.math.logsumexp(Qs_)
        return tt.sum(
            log_prob_actions[act, tt.arange(self.n_trials)]
        )  # PyMC makes it negative by default

    def _fit(
        self, pid, draws=1000, chains=2, cores=4, rerun=False, jupyter=False,
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
                    alpha = pm.Normal(f"alpha", 0.5, 0.25)
                    beta = pm.Gamma(f"beta", alpha=3, beta=1.5)
                    like = pm.DensityDist('like', self._theano_llik, observed=dict(_alpha=alpha, _beta=beta, _actions=actions, _rewards=rewards))
                    trace = pm.sample(start={'alpha': 0.5, 'beta': 3},
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
        if 'Q_left' in self.longform.columns:
            print('Qs already calculated')
            return
        else:
            print('Q table not calculated, running now')

        Q_table = pd.DataFrame()
        for pid in self.summary["PID"]:
            # fmt: off
            ## CALCULATE MEANS OF PARAMS: NEED TO CHANGE
            mean_alphas = np.vstack([
                self.traces[pid]['money'].posterior.alpha.to_numpy().mean(axis=1).mean(axis=0),
                self.traces[pid]['other'].posterior.alpha.to_numpy().mean(axis=1).mean(axis=0)
            ])
            
            actions = np.vstack([
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Action'].to_numpy().astype(int),
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Action'].to_numpy().astype(int)
            ])
            rewards = np.vstack([
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Reward'].to_numpy().astype(int),
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Reward'].to_numpy().astype(int)
            ])
            # fmt: on

            ## UPDATE VALUES: NEED TO UPDATE
            Qs = np.ones(shape=(actions.shape[0], actions.shape[1], 2)) * 0.5
            PEs = np.zeros(shape=(actions.shape[0], actions.shape[1]))
            for b, block in enumerate(["money", "other"]):
                block_actions = actions[b]
                block_rewards = rewards[b]

                for t, (act, rw) in enumerate(zip(block_actions, block_rewards)):
                    PEs[b, t] = rw - Qs[b, t - 1, act]
                    Qs[b, t, act] = Qs[b, t - 1, act] + mean_alphas[b] * PEs[b, t]
                    Qs[b, t, 1 - act] = Qs[b, t - 1, 1 - act]

            participant_Q_table = pd.DataFrame.from_dict(
                {
                    "PID": [pid] * actions.shape[1] * self.n_blocks,
                    "Trial": np.hstack(
                        [
                            np.arange(actions.shape[1]) + 1,
                            np.arange(actions.shape[1]) + 1,
                        ]
                    ),
                    "Q_left": np.hstack([Qs[0, :, 0], Qs[1, :, 0],]),
                    "Q_right": np.hstack([Qs[0, :, 1], Qs[1, :, 1],]),
                    "PE": np.hstack([PEs[0, :], PEs[1, :],]),
                    "Type": np.hstack(
                        [
                            np.array(["money"] * self.n_trials),
                            np.array(["other"] * self.n_trials),
                        ]
                    ),
                }
            )
            Q_table = pd.concat([Q_table, participant_Q_table])
        self.longform = self.longform.merge(Q_table, on=["PID", "Type", "Trial"])
        self.longform.to_csv(
            f"{self.save_path}/{self.model_name}/longform.csv", index=False
        )

    def _calc_lik(self, pid, block):
        alpha_values = self.traces[pid][block].posterior.alpha.values.flatten()
        beta_values = self.traces[pid][block].posterior.beta.values.flatten()
        actions = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].to_numpy().astype(int)
        rewards = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].to_numpy().astype(int)
        lik = []
        for alpha, beta in zip(alpha_values, beta_values):
            Qs = np.ones((len(actions) + 1, 2), dtype=float) * 0.5
            for t,(a,r) in enumerate(zip(actions, rewards)):
                delta = r - Qs[t, a]
                Qs[t + 1, a] = Qs[t, a] + alpha * delta
                Qs[t + 1, 1 - a] = Qs[t, 1 - a]
            # Apply the softmax transformation in a vectorized way to the values
            Qs_ = Qs * beta
            log_prob_actions = Qs_ - logsumexp(Qs_, axis=1)[:, None]
            log_prob_actions = log_prob_actions[np.arange(len(actions)), actions]
            lik.append(np.sum(log_prob_actions[1:]))
        return np.array(lik)