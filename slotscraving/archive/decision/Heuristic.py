## BAYESIAN LIBRARIES
from multiprocessing.sharedctypes import Value
import pymc3 as pm
import theano.tensor as tt
import theano
import arviz as az

## STANDARD LIBRARIES
import numpy as np
import pandas as pd

## UTILITIES
from IPython.display import clear_output
import os

class Heuristic(object):
    def __init__(self, model_name, save_path, summary=None, longform=None):
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

        self.model_description = 'Heuristic: switch after two losses'
        self.params = ['eps']
        self.model_name = model_name
        self.save_path = save_path

        if not os.path.exists(f'{save_path}/{model_name}/'):
            os.makedirs(f'{save_path}/{model_name}/')
        if not os.path.exists(f'{save_path}/{model_name}/traces/'):
            os.makedirs(f'{save_path}/{model_name}/traces/')
            
        if os.path.exists(f'{save_path}/{model_name}/params.csv'):
            params = pd.read_csv(f'{save_path}/{model_name}/params.csv')
            self.n_subj = summary.shape[0]
            self.n_trials = int(params['n_trials'])
            self.n_blocks = int(params['n_blocks'])
            self.n_options = int(params['n_options'])
            self.n_params = len(self.params)
            self.fit_complete = params['fit_complete'][0]

            # self.summary = pd.read_csv(f'{save_path}/{model_name}/summary.csv')
            # self.longform = pd.read_csv(f'{save_path}/{model_name}/longform.csv')
            self.summary = summary
            self.longform = longform

            self.traces = {}
            for pid in self.summary['PID']:
                try:
                    self.traces[pid] = az.from_netcdf(f'{save_path}/{model_name}/traces/{pid}.nc')
                except FileNotFoundError:
                    print(f'{pid} not run yet')

        else:
            if summary is None or longform is None:
                raise ValueError('Model has not been run, you need to specific summary and longform dfs.')
            
            self.n_subj = summary.shape[0]
            self.n_trials = 60
            self.n_blocks = 2
            self.n_options = 2
            self.n_params = len(self.params)
            self.fit_complete = False

            self.summary = summary
            self.longform = longform
            self.traces = {}

    # DETERMINE LIKELIHOOD OF PARAMETER
    def update_func(self, action, reward, Qs, alpha, n_blocks):
        """This function updates the Q table according to the RL update rule. 
        It will be called by theano.scan to do so recursively, given the observed data and the alpha parameter
        THIS LIKELY NEEDS TO BE UDPATED FOR EVERY MODEL.

        Args:
            action (theano.tensor.imatrix): Theano matrix of actions for each block
            reward (theano.tensor.imatrix): Theano matrix of rewards for each block
            Qs (theano.tensor.fmatrix): Theano matrix of values for each block and choice
            alpha (theano.tensor.fvector): Theano vector of alphas for each block
            n_blocks (theano.tensor.iscalar): Theano scalar for number of blocks

        Returns:
            theano.tensor.fmatrix: Updated matrix of values
        """
        PE = reward - Qs[tt.arange(n_blocks), action]
        Qs = tt.set_subtensor(
            Qs[tt.arange(n_blocks), action],
            Qs[tt.arange(n_blocks), action] + alpha * PE,
        )
        return Qs

    ## Theano-PyMC wrapper to calculate Qs for actions and rewards
    def _theano_llik(self, _eps, _strat):
        """Wrapper for calculating likelihood.

        Args:
            _eps (theano.tensor.fvector): Theano vector of randomness for each block
            _actions (theano.tensor.imatrix): Theano matrix of actions for each block

        Returns:
            theano.tensor.iscalar: Log-likelihood of chosen parameters
        """
        strat = tt.cast(_strat.T, 'int16').eval()
        eps = tt.cast(_eps, 'float64')
        return tt.sum(
            np.log(
                (strat==0)*eps.reshape(-1,1) + 
                (strat==1)*(1-eps).reshape(-1,1)
            )
        )

    # DISCOVER PARAMETER BY MAXIMIZING LOG LIKELIHOOD
    def fit(self, draws=500, chains=2, cores=4, jupyter=False, rerun=False):
        """Loop over all subjects in the longform dataframe to fit with pm.DensityDist.
        THIS FUNCTION LIKELY NEEDS TO BE UPDATED FOR EVERY MODEL. The priors need to be updated
        during the 'with pm.Model() as m:' block.

        Args:
            draws (int, optional): Number of draws in pm.Sample. Defaults to 500.
            chains (int, optional): Number of chains in pm.Sample. Defaults to 2.
            cores (int, optional): Number of cores used for sampling. Defaults to 4.
            jupyter (bool, optional): Set to True if model is being run in a Jupyter notebook. Defaults to False.
        """

        pd.DataFrame({
            'model_description': self.model_description,
            'params': self.params,
            'n_subj': self.n_subj,
            'n_trials': self.n_trials,
            'n_blocks': self.n_blocks,
            'n_options': self.n_options,
            'fit_complete': self.fit_complete,
        }, index=[0]).to_csv(f'{self.save_path}/{self.model_name}/params.csv', index=False)

        self.summary.to_csv(f'{self.save_path}/{self.model_name}/summary.csv', index=False)
        self.longform.to_csv(f'{self.save_path}/{self.model_name}/longform.csv', index=False)

        # self.traces = {}

        # fmt: off
        for i, pid in enumerate(self.summary['PID']):
            if jupyter:
                clear_output(wait=True)

            if os.path.exists(f'{self.save_path}/{self.model_name}/traces/{pid}.nc') and not rerun:
                print(f'Participant {i+1} completed...')
                # self.traces[pid] = az.from_netcdf(f'{self.save_path}/{self.model_name}/traces/{pid}.nc')
                continue

            print(f'Running participant {i+1} of {self.n_subj}')
            actions = np.vstack([
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Action'].to_numpy().astype(int),
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Action'].to_numpy().astype(int)
            ])
            rewards = np.vstack([
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Reward'].to_numpy().astype(int),
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Reward'].to_numpy().astype(int)
            ])
            strat = np.zeros((actions.shape[0], actions.shape[1]-2))
            for i in np.arange(actions.shape[1]):
                if i<2:
                    continue
                should_switch = np.all(np.array([rewards[:,i-2]==rewards[:,i-1], rewards[:,i-1]==0]), axis=0)
                do_switch = actions[:, i-1]!=actions[:, i]
                strat[:,i-2] = should_switch==do_switch
            strat = strat.astype(int)
        #fmt: on

            with pm.Model() as m:
                eps = pm.Uniform(f"eps", 0, 1, shape=self.n_blocks)
                like = pm.DensityDist('like', self._theano_llik, observed=dict(_eps=eps, _strat=strat))
                self.traces[pid] = pm.sample(start={'eps': np.array([0.2, 0.2])},
                    draws=draws, chains=chains, cores=cores, return_inferencedata=True, idata_kwargs={"density_dist_obs": False}
                )
            self.traces[pid].to_netcdf(f'{self.save_path}/{self.model_name}/traces/{pid}.nc')
            
        self.fit_complete = True

    def calc_Q_table(self):
        """This function calculates mean alphas across traces for each parameter, and recalculates and stores Qs
        based on mean estimated parameter values. THIS FUNCTION LIKELY NEEDS TO BE UPDATED FOR EACH MODEL.
        """
        Q_table = pd.DataFrame()
        for pid in self.summary['PID']:

            ## CALCULATE MEANS OF PARAMS: NEED TO CHANGE
            mean_alphas = self.traces[pid].posterior.eps.to_numpy().mean(axis=1).mean(axis=0)
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

    def calc_bics(self):
        self.bics = {}
        for pid in self.summary['PID']:
            self.bics[pid] = []
            actions = np.vstack([
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Action'].to_numpy().astype(int),
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Action'].to_numpy().astype(int),
                # self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='numberbar_mixed')]['Action'].to_numpy().astype(int)
            ])
            rewards = np.vstack([
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='money')]['Reward'].to_numpy().astype(int),
                self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='other')]['Reward'].to_numpy().astype(int),
                # self.longform[(self.longform['PID']==pid) & (self.longform['Type']=='numberbar_mixed')]['Reward'].to_numpy().astype(int)
            ])
            strat = np.zeros((actions.shape[0], actions.shape[1]-2))
            for i in np.arange(actions.shape[1]):
                if i<2:
                    continue
                should_switch = np.all(np.array([rewards[:,i-2]==rewards[:,i-1], rewards[:,i-1]==0]), axis=0)
                do_switch = actions[:, i-1]!=actions[:, i]
                strat[:,i-2] = should_switch==do_switch
            strat = strat.astype(int)

            for eps_chain in self.traces[pid].posterior.eps.values:
                for block_params in eps_chain:
                    sample_bic = []
                    for b, block in enumerate(['money', 'other']):
                        block_param = block_params[b]
                        block_strat = strat[b]
                        log_prob_actions = np.zeros(len(block_strat))
                        lpa = np.log([block_param, 1-block_param])
                        for t, a in enumerate(block_strat):
                            log_prob_actions[t] = lpa[a]
                        lik = -np.sum(log_prob_actions[1:])
                        bic = np.log(len(block_strat)) * self.n_params + 2 * lik
                        sample_bic.append(bic)
                    self.bics[pid].append(sample_bic)
            self.bics[pid] = np.array(self.bics[pid])

    def divergences(self):
        divergences = []
        for pid in self.summary['PID']:
            div = self.traces[pid].sample_stats.diverging.to_numpy().sum()
            if div > 40:
                divergences.append([pid, div, 'Major Divergences'])
            elif div > 10:
                divergences.append([pid, div, 'Minor'])
        divergences = np.array(divergences)
        if divergences.size>0:
            return pd.DataFrame.from_dict({
                'PID': divergences[:,0],
                'Divergences': divergences[:,1],
                'Level': divergences[:,2]
            })
        else:
            return 'No divergences'