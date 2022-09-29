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

from standard.Prototype import Prototype


class EVRPE0step(Prototype):
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
        self.craving_model_name = "EVRPE0step"
        self.equation = "w0 + w1 * Q - w2 * PE"
        self.n_params = 4
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

    def fit(self, pid, draws=1000, cores=4):

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
            w2_mean = pm.Normal('w2_mean', mu=0, sigma=1, shape=self.n_blocks)
            # sample_sd = pm.Exponential('sample_sd', lam=2, shape=n_blocks)
            
            w0 = pm.Normal('w0', mu=w0_mean, sigma=0.5, shape=(self.n_samples, self.n_blocks))
            w1 = pm.Normal('w1', mu=w1_mean, sigma=0.5, shape=(self.n_samples, self.n_blocks))
            w2 = pm.Normal('w2', mu=w2_mean, sigma=0.5, shape=(self.n_samples, self.n_blocks))
            
            craving_sig = pm.Exponential('craving_sig', lam=1, shape=self.n_blocks)
            craving_sig_ = tt.reshape(tt.repeat(craving_sig, self.n_samples), (self.n_blocks, self.n_samples))
            craving_sig_ = tt.reshape(tt.repeat(craving_sig_, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))

            w0_ = tt.reshape(tt.repeat(w0.T, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))
            w1_ = tt.reshape(tt.repeat(w1.T, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))
            w2_ = tt.reshape(tt.repeat(w2.T, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))

            q_reg = tt.zeros((self.n_blocks, self.n_samples, norm_craving_ratings.shape[2]))
            pe_reg = tt.zeros((self.n_blocks, self.n_samples, norm_craving_ratings.shape[2]))
            # fmt: on

            for i, ind in enumerate(craving_inds):
                q_reg = tt.set_subtensor(q_reg[:, :, i], qs[:, :, ind])
                pe_reg = tt.set_subtensor(pe_reg[:, :, i], pes[:, :, ind])
            pred_craving = w0_ + w1_ * q_reg - w2_ * pe_reg
            # pred_craving = tt.mul(w1, q_reg)
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
                trace=[w0_mean, w1_mean, w2_mean, craving_sig],
            )

        self.traces[pid].to_netcdf(
            f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
        )

    # ImmEVRPE specific function to get sample predictions for a participant
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
        w2_samples = np.vstack(
            [
                self.traces[pid].posterior.w2_mean.values[0, :, :],
                self.traces[pid].posterior.w2_mean.values[1, :, :],
            ]
        )

        sample_preds = []

        for samp, (w0, w1, w2) in enumerate(zip(w0_samples, w1_samples, w2_samples)):

            w0_ = np.reshape(
                np.repeat(w0.T, norm_craving_ratings.shape[2]), (self.n_blocks, -1)
            )
            w1_ = np.reshape(
                np.repeat(w1.T, norm_craving_ratings.shape[2]), (self.n_blocks, -1)
            )
            w2_ = np.reshape(
                np.repeat(w2.T, norm_craving_ratings.shape[2]), (self.n_blocks, -1)
            )

            qs_reg = np.zeros((self.n_blocks, norm_craving_ratings.shape[2]))
            pes_reg = np.zeros((self.n_blocks, norm_craving_ratings.shape[2]))

            for b, block in enumerate(["money", "other"]):
                block_qs = qs[b, samp, :]
                block_pes = pes[b, samp, :]
                for i, ind in enumerate(craving_inds):
                    qs_reg[b, i] = block_qs[ind]
                    pes_reg[b, i] = block_pes[ind]
            pred = w0_ + w1_ * qs_reg - w2_ * pes_reg
            sample_preds.append(pred)

        sample_preds = np.array(sample_preds)

        return sample_preds

