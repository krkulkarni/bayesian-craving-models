# Graphing libraries
import arviz as az

# Bayesian libraries
import pymc3 as pm
import theano.tensor as tt
from theano import scan

# Standard libraries
import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from IPython.display import clear_output

from .Prototype import Prototype


class RPE1step(Prototype):
    def __init__(
        self,
        longform,
        decision_path,
        craving_path,
        model_name,
        n_samples=2000,
        n_trials=60,
        n_blocks=2,
    ):
        ##------- MODEL PARAMETERS -------##
        self.craving_model_name = "RPE1step"
        self.equation = "w0 + w1 * PE"
        self.n_params = 3
        ##--------------------------------##
        ###############################################################################

        super().__init__(
            longform,
            decision_path,
            craving_path,
            model_name,
            n_samples=n_samples,
            n_trials=n_trials,
            n_blocks=n_blocks,
        )

    def _fit(self, pid, cores=4, draws=1000):

        ## SPECIFY PYMC MODEL
        for block in ["money", "other"]:
            if pid not in self.traces.keys():
                self.traces[pid] = {}
            if block not in self.traces[pid].keys():
                self.traces[pid][block] = {}
            # fmt: off
            if os.path.exists(f"{self.craving_path}/{self.craving_model_name}/traces/{pid}_{block}.nc"):
                self.traces[pid][block] = az.from_netcdf(f"{self.craving_path}/{self.craving_model_name}/traces/{pid}_{block}.nc")
                continue
            # fmt: on
            else:
                if self.model_name=='rwrl':
                    (
                        pid,
                        actions,
                        rewards,
                        norm_craving_ratings,
                        norm_factor,
                        craving_inds,
                        qs,
                        pes,
                    ) = self._get_rwrl_actrewrate_qspes(pid, block)
                elif self.model_name=='rw':
                    (
                        pid,
                        actions,
                        rewards,
                        norm_craving_ratings,
                        norm_factor,
                        craving_inds,
                        qs,
                        pes,
                    ) = self._get_rw_actrewrate_qspes(pid, block)

                if norm_factor == 0:
                    print(f"No variation in ratings: {pid}_{block}")
                    self.traces[pid][block] = 'No variation in ratings'
                    continue

                try:
                    with pm.Model() as model:
                        # fmt: off
                        w0_mean = pm.Normal('w0_mean', mu=0, sigma=1)
                        w1_mean = pm.Normal('w1_mean', mu=0, sigma=1)
                        # sample_sd = pm.Exponential('sample_sd', lam=2)
                        
                        w0 = pm.Normal('w0', mu=w0_mean, sigma=0.5, shape=self.n_samples)
                        w1 = pm.Normal('w1', mu=w1_mean, sigma=0.5, shape=self.n_samples)
                        
                        craving_sig = pm.Exponential('craving_sig', lam=1)

                        pe_reg = tt.zeros((self.n_samples, norm_craving_ratings.shape[1]))

                        for i, ind in enumerate(craving_inds):
                            pe_reg = tt.set_subtensor(pe_reg[:, i], pes[:, ind] + pes[:, ind - 1])
                            pred_craving = w0 + w1 * pe_reg.T
                        pred = pm.Normal(
                            "pred",
                            mu=pred_craving.T,
                            sigma=craving_sig,
                            observed=norm_craving_ratings,
                        )
                        self.traces[pid][block] = pm.sample(
                            draws=draws,
                            chains=2,
                            cores=cores,
                            return_inferencedata=True,
                            trace=[w0_mean, w1_mean, craving_sig],
                        )

                    self.traces[pid][block].to_netcdf(
                        f"{self.craving_path}/{self.craving_model_name}/traces/{pid}_{block}.nc"
                    )
                except RuntimeError:
                    print(f"{pid}_{block} failed to fit {self.craving_model_name}")
                    self.skipped_list.append(f"{pid}_{block}")
            # fmt: on

    # Specific function to get sample predictions for a participant
    def get_sample_predictions(
        self, pid, block, actions, rewards, norm_craving_ratings, craving_inds, qs, pes
    ):
        try:
            w0_samples = np.hstack(
                [
                    self.traces[pid][block].posterior.w0_mean.values[0, :],
                    self.traces[pid][block].posterior.w0_mean.values[1, :],
                ]
            )
            w1_samples = np.hstack(
                [
                    self.traces[pid][block].posterior.w1_mean.values[0, :],
                    self.traces[pid][block].posterior.w1_mean.values[1, :],
                ]
            )
        except AttributeError:
            print(
                f"No posterior: {pid}_{block} failed to fit {self.craving_model_name}"
            )
            return None

        sample_preds = []

        for samp, (w0, w1) in enumerate(zip(w0_samples, w1_samples)):

            pes_reg = np.zeros(norm_craving_ratings.shape[1])

            pes_ = pes[samp, :]
            for i, ind in enumerate(craving_inds):
                pes_reg[i] = pes_[ind] + pes_[ind - 1]
            pred = w0 + w1 * pes_reg
            sample_preds.append(pred)

        sample_preds = np.array(sample_preds)

        return sample_preds

