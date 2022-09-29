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


class Biased(Prototype):
    def __init__(self, model_name, save_path, load=False, summary=None, longform=None):
        """Biased model class. Accepts the summary and longform dataframes as inputs. 
        Can also load from netcdf folder, as long as param, summary, longform csvs, and 
        netcdf model traces are available. Likely DOES NOT NEED TO GET MODIFIED beyond model name and params.

        Args:
            summary (pd.DataFrame, optional): Summary dataframe for subjects. Defaults to None.
            longform (pd.DataFrame, optional): Longform data for all subjects. Defaults to None.
            load_from (str, optional): Path to saved netcdf folder. Defaults to None.
            model_name (str, optional): Description of model. Defaults to None.

        Raises:
            ValueError: Raise error if model name is not given.
        """

        self.model_description = "Biased model"
        self.params = ["bias"]
        super().__init__(
            model_name=model_name,
            save_path=save_path,
            load=load,
            summary=summary,
            longform=longform,
        )

    ## Theano-PyMC wrapper to calculate Qs for actions and rewards
    def _theano_llik(self, _bias, _actions):
        """Wrapper for calculating likelihood.

        Args:
            _bias (theano.tensor.fscalar): Theano scalar for bias
            _actions (theano.tensor.ivector): Theano vector of action

        Returns:
            theano.tensor.fscalar: Log-likelihood of chosen parameters
        """
        act = tt.cast(_actions, "int16").eval()
        bias = tt.cast(_bias, "float64")
        return tt.sum(np.log((act == 0) * bias + (act == 1) * (1 - bias)))

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
                # rewards = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].to_numpy().astype(int)
                with pm.Model() as m:
                    bias = pm.Normal(f"bias", 0.5, 0.2)
                    like = pm.DensityDist('like', self._theano_llik, observed=dict(_bias=bias, _actions=actions))
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
        # fmt: on

    def calc_Q_table(self):
        print("No Q values available for this model.")

    def _calc_lik(self, pid, block):
        bias_values = self.traces[pid][block].posterior.bias.values.flatten()
        actions = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].to_numpy().astype(int)
        rewards = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].to_numpy().astype(int)
        lik = []
        for bias in bias_values:
            log_prob_actions = np.zeros(len(actions))
            lpa = np.log([bias, 1-bias])
            for t,(a,r) in enumerate(zip(actions, rewards)):
                log_prob_actions[t] = lpa[a]
            lik.append(np.sum(log_prob_actions))
        return np.array(lik)