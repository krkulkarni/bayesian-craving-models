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

from .Prototype import Prototype

class Heuristic(Prototype):
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

        self.model_description = 'Heuristic: switch after two losses'
        self.params = ['eps']
        super().__init__(
            model_name=model_name,
            save_path=save_path,
            load=load,
            summary=summary,
            longform=longform,
        )

    ## Theano-PyMC wrapper to calculate Qs for actions and rewards
    def _theano_llik(self, _eps, _strat):
        """Wrapper for calculating likelihood.

        Args:
            _eps (theano.tensor.fvector): Theano vector of randomness for each block
            _actions (theano.tensor.imatrix): Theano matrix of actions for each block

        Returns:
            theano.tensor.iscalar: Log-likelihood of chosen parameters
        """
        strat = tt.cast(_strat, "int16").eval()
        eps = tt.cast(_eps, "float64")
        return tt.sum(np.log((strat == 0) * eps + (strat == 1) * (1 - eps)))

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

                strat = np.zeros(len(actions))
                for i, a in enumerate(actions):
                    if i < 2:
                        continue
                    should_switch = np.all(np.array([rewards[i-2]==rewards[i-1], rewards[i-1]==0]), axis=0)
                    do_switch = actions[i-1]!=actions[i]
                    strat[i] = should_switch==do_switch
                strat = strat.astype(int)[2:]

                with pm.Model() as m:
                    eps = pm.Uniform(f"bias", 0, 1)
                    like = pm.DensityDist('like', self._theano_llik, observed=dict(_eps=eps, _strat=strat))
                    trace = pm.sample(
                        start={'bias': 0.5}, draws=draws, chains=chains, cores=cores, 
                        return_inferencedata=True, idata_kwargs={"density_dist_obs": False}
                    )
                trace.to_netcdf(f'{self.save_path}/{self.model_name}/traces/{pid}_{block}.nc')
                print(f'Participant {pid}, {block} block completed...')
            except RuntimeError:
                print(f'Participant {pid}, {block} block failed...')
                if (pid, block) not in self.skip_list:
                    self.skip_list.append((pid, block))
                continue

    def calc_Q_table(self):
        print("No Q values available for this model.")

    def _calc_lik(self, pid, block):
        eps_values = self.traces[pid][block].posterior.bias.values.flatten()
        actions = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].to_numpy().astype(int)
        rewards = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].to_numpy().astype(int)
        strat = np.zeros(len(actions))
        for i, a in enumerate(actions):
            if i < 2:
                continue
            should_switch = np.all(np.array([rewards[i-2]==rewards[i-1], rewards[i-1]==0]), axis=0)
            do_switch = actions[i-1]!=actions[i]
            strat[i] = should_switch==do_switch
        strat = strat.astype(int)[2:]

        lik = []
        for eps in eps_values:
            log_prob_actions = [np.log(eps), np.log(eps)]
            lpa = np.log([eps, 1-eps])
            for t, a in enumerate(strat):
                log_prob_actions.append(lpa[a])
            lik.append(np.sum(log_prob_actions))
        return np.array(lik)

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