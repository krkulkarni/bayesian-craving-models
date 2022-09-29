# Graphing libraries
import arviz as az

# Bayesian libraries
import pymc3 as pm
import theano.tensor as tt
from theano import scan

# Standard libraries
import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from IPython.display import clear_output

from standard.Prototype import Prototype


class CueElic(Prototype):
    def __init__(
        self,
        longform,
        decision_path,
        craving_path,
        model_name,
        n_samples=1000,
        n_trials=60,
        n_blocks=2,
    ):
        ##------- MODEL PARAMETERS -------##
        self.craving_model_name = "CueElic"
        self.equation = "w0 + w1 * cueelic"
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

    def fit(self, pid, cores=4, draws=1000):

        (
            pid,
            actions,
            rewards,
            norm_craving_ratings,
            norm_factor,
            craving_inds,
            qs,
            pes,
        ) = self._get_rw_actrewrate_qspes(pid)

        ## SPECIFY PYMC MODEL
        with pm.Model() as model:
            # fmt: off
            w0_mean = pm.Normal('w0_mean', mu=0, sigma=1, shape=self.n_blocks)
            w1_mean = pm.Normal('w1_mean', mu=0, sigma=1, shape=self.n_blocks)
            
            w0 = pm.Normal('w0', mu=w0_mean, sigma=0.5, shape=(self.n_samples, self.n_blocks))
            w1 = pm.Normal('w1', mu=w1_mean, sigma=0.5, shape=(self.n_samples, self.n_blocks))
            
            craving_sig = pm.Exponential('craving_sig', lam=1, shape=self.n_blocks)
            craving_sig_ = tt.reshape(tt.repeat(craving_sig, self.n_samples), (self.n_blocks, self.n_samples))
            craving_sig_ = tt.reshape(tt.repeat(craving_sig_, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))

            w0_ = tt.reshape(tt.repeat(w0.T, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))
            w1_ = tt.reshape(tt.repeat(w1.T, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))

            cueelic_reg = tt.zeros((self.n_blocks, self.n_samples, norm_craving_ratings.shape[2]))
            # fmt: on

            for i, ind in enumerate(craving_inds):
                cueelic_reg = tt.set_subtensor(
                    cueelic_reg[:, :, i],
                    np.repeat([rewards[:, ind]], self.n_samples, axis=0).T,
                )

            pred_craving = w0_ + w1_ * cueelic_reg
            pred = pm.Normal(
                "pred",
                mu=pred_craving,
                sigma=craving_sig_,
                observed=norm_craving_ratings,
            )
            self.traces[pid] = pm.sample(
                draws=draws,
                chains=2,
                cores=cores,
                return_inferencedata=True,
                trace=[w0_mean, w1_mean, craving_sig],
            )

        self.traces[pid].to_netcdf(
            f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
        )

    def get_sample_predictions(
        self, pid, actions, rewards, norm_craving_ratings, craving_inds, qs, pes
    ):
        w0_samples = np.vstack(
            [
                self.traces[pid].posterior.w0_mean.values[0, :, :],
                self.traces[pid].posterior.w0_mean.values[1, :, :],
            ]
        )
        w1_samples = np.vstack(
            [
                self.traces[pid].posterior.w1_mean.values[0, :, :],
                self.traces[pid].posterior.w1_mean.values[1, :, :],
            ]
        )

        sample_preds = []
        for w0, w1 in zip(w0_samples, w1_samples):

            w0_ = np.reshape(
                np.repeat(w0.T, norm_craving_ratings.shape[2]), (self.n_blocks, -1)
            )
            w1_ = np.reshape(
                np.repeat(w1.T, norm_craving_ratings.shape[2]), (self.n_blocks, -1)
            )

            cueelic_reg = rewards[:, craving_inds]
            pred = w0_ + w1_ * cueelic_reg
            sample_preds.append(pred)
        sample_preds = np.array(sample_preds)

        return sample_preds
