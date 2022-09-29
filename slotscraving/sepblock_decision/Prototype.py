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

# DETERMINE LIKELIHOOD OF PARAMETER
def update_func():
    raise NotImplementedError


class Prototype(object):
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

        self.model_name = model_name
        self.save_path = save_path

        if not os.path.exists(f"{save_path}/{model_name}/"):
            os.makedirs(f"{save_path}/{model_name}/")
        if not os.path.exists(f"{save_path}/{model_name}/traces/"):
            os.makedirs(f"{save_path}/{model_name}/traces/")

        if load:
            dh = display('Loading completed model...', display_id=True)
            if (summary is None and longform is None):
                try:
                    self.summary = pd.read_csv(f"{save_path}/{model_name}/summary.csv")
                    self.longform = pd.read_csv(
                        f"{save_path}/{model_name}/longform.csv"
                    )
                except FileNotFoundError:
                    raise FileNotFoundError("No summary or longform data found.")
            else:
                self.summary = summary
                self.longform = longform

            params = pd.read_csv(f"{save_path}/{model_name}/params.csv")
            self.n_subj = self.summary.shape[0]
            self.n_trials = int(params["n_trials"])
            self.n_blocks = int(params["n_blocks"])
            self.n_options = int(params["n_options"])
            self.n_params = len(self.params)
            # self.fit_complete = params['fit_complete'][0]

            self.traces = {}
            for i, pid in enumerate(self.summary["PID"]):
                self.traces[pid] = {}
                dh.update(f'Loading traces for participant {i+1} of {len(self.summary["PID"])}: {pid}')
                for block in ["money", "other"]:
                    try:
                        self.traces[pid][block] = az.from_netcdf(
                            f"{save_path}/{model_name}/traces/{pid}_{block}.nc"
                        )
                    except FileNotFoundError:
                        print(f"{pid} not run yet")

            if os.path.exists(f"{save_path}/{model_name}/skip_list.csv"):
                self.skip_list = list(
                    np.loadtxt(f"{save_path}/{model_name}/skip_list.csv")
                )
            else:
                self.skip_list = []

        else:
            if summary is None or longform is None:
                raise ValueError(
                    "Model has not been run, you need to specific summary and longform dfs."
                )

            self.n_subj = summary.shape[0]
            self.n_trials = 60
            self.n_blocks = 2
            self.n_options = 2
            self.n_params = len(self.params)
            self.fit_complete = False

            self.summary = summary
            self.longform = longform
            self.traces = {}
            self.skip_list = []

    ## Theano-PyMC wrapper to calculate Qs for actions and rewards
    def _theano_llik(self, _alpha, _beta, _actions, _rewards):
        raise NotImplementedError

    def _fit(self, pid, draws=1000, chains=2, cores=4, rerun=False, jupyter=False):
        raise NotImplementedError

    def calc_decision_model(
        self, draws=1000, chains=2, cores=4, jupyter=False, rerun=False
    ):
        """Loop over all subjects in the longform dataframe to fit with pm.DensityDist.
        THIS FUNCTION LIKELY NEEDS TO BE UPDATED FOR EVERY MODEL. The priors need to be updated
        during the 'with pm.Model() as m:' block.

        Args:
            draws (int, optional): Number of draws in pm.Sample. Defaults to 500.
            chains (int, optional): Number of chains in pm.Sample. Defaults to 2.
            cores (int, optional): Number of cores used for sampling. Defaults to 4.
            jupyter (bool, optional): Set to True if model is being run in a Jupyter notebook. Defaults to False.
        """
        # fmt: off
        if jupyter:
            dh = display('Starting decision modeling...', display_id=True)
        pd.DataFrame({
            'model_description': self.model_description,
            'n_subj': self.n_subj,
            'n_trials': self.n_trials,
            'n_blocks': self.n_blocks,
            'n_options': self.n_options,
            'fit_complete': self.fit_complete,
        }, index=[0]).to_csv(f'{self.save_path}/{self.model_name}/params.csv', index=False)

        self.summary.to_csv(f'{self.save_path}/{self.model_name}/summary.csv', index=False)
        self.longform.to_csv(f'{self.save_path}/{self.model_name}/longform.csv', index=False)
        # fmt: on

        for i, pid in enumerate(self.summary["PID"]):
            if jupyter:
                clear_output(wait=True)
                dh.display(f"Participant {i+1} of {self.n_subj}")
            else:
                print(f"Participant {i+1} of {self.n_subj}")
            self._fit(
                pid=pid,
                draws=draws,
                chains=chains,
                cores=cores,
                rerun=rerun,
                jupyter=jupyter,
            )
        if jupyter:
            dh.update(f"Done! Skipped {self.skip_list}")
        np.savetxt(f"{self.save_path}/{self.model_name}/skip_list.csv", self.skip_list)

    def calc_Q_table(self):
        raise NotImplementedError

    def calc_ics(self):
        self.ics = {}

        # fmt: off
        for pid in self.summary["PID"]:
            self.ics[pid] = {}
            for block in ["money", "other"]:
                try:
                    self.ics[pid][block] = {}
                    self.ics[pid][block]["likelihood"] = self._calc_lik(pid, block)
                    # self.ics[pid][block]["likelihood"] = self.traces[pid][block].log_likelihood.like.to_numpy().flatten()
                    self.ics[pid][block]["bic"] = np.log(self.n_trials) * self.n_params - 2 * self.ics[pid][block]["likelihood"]
                    pdic = 2*(self.ics[pid][block]["likelihood"] - self.ics[pid][block]['likelihood'].mean())
                    self.ics[pid][block]["dic"] = -2*self.ics[pid][block]["likelihood"] + 2*pdic
                except KeyError:
                    print(f"{pid}_{block} not run yet")
                    continue
        # fmt: on

    # def divergences(self):
    #     divergences = []
    #     for pid in self.summary["PID"]:
    #         div = self.traces[pid].sample_stats.diverging.to_numpy().sum()
    #         if div > 40:
    #             divergences.append([pid, div, "Major Divergences"])
    #         elif div > 10:
    #             divergences.append([pid, div, "Minor"])
    #     divergences = np.array(divergences)
    #     if divergences.size > 0:
    #         return pd.DataFrame.from_dict(
    #             {
    #                 "PID": divergences[:, 0],
    #                 "Divergences": divergences[:, 1],
    #                 "Level": divergences[:, 2],
    #             }
    #         )
    #     else:
    #         return pd.DataFrame.from_dict({"PID": [], "Divergences": [], "Level": []})

