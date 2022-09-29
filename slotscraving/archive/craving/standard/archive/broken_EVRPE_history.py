# Graphing libraries
from calendar import c
import arviz as az

# Bayesian libraries
import pymc3 as pm
import theano.tensor as tt
from theano import scan

# Standard libraries
import numpy as np
import pandas as pd
import os
from IPython.display import clear_output


class EVRPE(object):
    def __init__(
        self, longform, decision_path, craving_path, model_name, n_samples=100
    ):
        self.craving_model_name = "EVRPE"

        self.longform = longform
        self.num_participants = len(longform["PID"].unique())

        self.n_blocks = 2
        self.n_samples = n_samples
        self.n_trials = 60

        self.decision_path = decision_path
        self.model_name = model_name

        self.craving_path = craving_path
        if not os.path.exists(f"{self.craving_path}/{self.craving_model_name}/"):
            os.makedirs(f"{self.craving_path}/{self.craving_model_name}/")
        if not os.path.exists(f"{self.craving_path}/{self.craving_model_name}/traces/"):
            os.makedirs(f"{self.craving_path}/{self.craving_model_name}/traces/")

        pd.DataFrame.from_dict(
            {
                "craving_model_name": [self.craving_model_name],
                "n_blocks": [self.n_blocks],
                "num_participants": [self.num_participants],
                "n_samples": [self.n_samples],
                "n_trials": [self.n_trials],
                "decision_model_name": [model_name],
            }
        ).to_csv(f"{self.craving_path}/{self.craving_model_name}/params.csv")

        self.traces = {}

    def _get_actrewrate_qspes(self, pid):
        ## RETRIEVE ACTIONS, REWARDS, CRAVING RATINGS
        # fmt: off
        actions = np.vstack([
            self.longform[(self.longform['PID']==pid)&(self.longform['Type']=='money')]['Action'].values,
            self.longform[(self.longform['PID']==pid)&(self.longform['Type']=='other')]['Action'].values
        ])
        rewards = np.vstack([
            self.longform[(self.longform['PID']==pid)&(self.longform['Type']=='money')]['Reward'].values,
            self.longform[(self.longform['PID']==pid)&(self.longform['Type']=='other')]['Reward'].values
        ])
        craving_ratings = np.vstack([
            self.longform[(self.longform['PID']==pid)&(self.longform['Type']=='money')]['Craving Rating'].values,
            self.longform[(self.longform['PID']==pid)&(self.longform['Type']=='other')]['Craving Rating'].values
        ])
        # fmt: on

        ## NORMALIZE CRAVING RATINGS
        craving_inds = np.nonzero(craving_ratings[0] > -1)[0]
        craving_inds = np.delete(craving_inds, [12])

        craving_ratings = np.tile(
            craving_ratings[:, craving_inds], self.n_samples
        ).reshape(self.n_blocks, self.n_samples, -1)
        norm_craving_ratings = (craving_ratings - 25) / 25

        ## RETRIEVE ALPHAS FROM DECISION MODEL
        traces = az.from_netcdf(
            f"{self.decision_path}/{self.model_name}/traces/{pid}.nc"
        )
        alphas = np.vstack(
            [traces.posterior.alpha.values[0], traces.posterior.alpha.values[1]]
        )[: self.n_samples, :].T

        ## GENERATE QS AND PES
        qs = np.zeros((self.n_blocks, self.n_samples, self.n_trials))
        pes = np.zeros((self.n_blocks, self.n_samples, self.n_trials))

        for b in np.arange(self.n_blocks):
            block_actions = actions[b]
            block_rewards = rewards[b]
            block_alphas = alphas[b]
            for i in np.arange(self.n_samples):
                al = block_alphas[i]
                qs_ = np.array([0.5, 0.5])
                chosen_qs = [0.5]
                chosen_pes = []
                for a, r in zip(block_actions, block_rewards):
                    pe = r - qs_[a]
                    chosen_pes.append(pe)
                    qs_[a] = qs_[a] + al * pe
                    chosen_qs.append(qs_[a])
                qs[b, i, :] = chosen_qs[:-1]
                pes[b, i, :] = chosen_pes

        return actions, rewards, norm_craving_ratings, craving_inds, qs, pes

    def fit(self, pid):

        (
            actions,
            rewards,
            norm_craving_ratings,
            craving_inds,
            qs,
            pes,
        ) = self._get_actrewrate_qspes(pid)

        ## SPECIFY PYMC MODEL
        with pm.Model() as model:
            # fmt: off
            # logit_ff = pm.Normal('logit_ff', mu=0, sigma=2)
            # ff = pm.math.invlogit(logit_ff)
            ff = pm.Uniform('ff', lower=0, upper=1)
            w0_mean = pm.Normal('w0_mean', mu=0, sigma=1, shape=self.n_blocks)
            w1_mean = pm.Normal('w1_mean', mu=0, sigma=1, shape=self.n_blocks)
            w2_mean = pm.Normal('w2_mean', mu=0, sigma=1, shape=self.n_blocks)
            # sample_sd = pm.Exponential('sample_sd', lam=2, shape=n_blocks)
            
            w0 = pm.Normal('w0', mu=w0_mean, sigma=0.5, shape=(self.n_samples, self.n_blocks))
            w1 = pm.Normal('w1', mu=w1_mean, sigma=0.5, shape=(self.n_samples, self.n_blocks))
            w2 = pm.Normal('w2', mu=w2_mean, sigma=0.5, shape=(self.n_samples, self.n_blocks))
            
            craving_sig = pm.Exponential('craving_sig', lam=1)

            w0_ = tt.reshape(tt.repeat(w0.T, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))
            w1_ = tt.reshape(tt.repeat(w1.T, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))
            w2_ = tt.reshape(tt.repeat(w2.T, norm_craving_ratings.shape[2]), (self.n_blocks, self.n_samples, -1))

            q_reg = tt.zeros((self.n_blocks, self.n_samples, norm_craving_ratings.shape[2]))
            pe_reg = tt.zeros((self.n_blocks, self.n_samples, norm_craving_ratings.shape[2]))
            # fmt: on

            for i, ind in enumerate(craving_inds):
                result, _ = scan(
                    fn=lambda prior_result, ff: prior_result * ff,
                    outputs_info=tt.ones(1),
                    non_sequences=ff,
                    n_steps=ind,
                )

                ff_vec = tt.reshape(
                    tt.tile(
                        tt.reshape(
                            tt.tile(result.flatten(), self.n_samples),
                            (self.n_samples, -1),
                        ),
                        self.n_blocks,
                    ),
                    (self.n_blocks, self.n_samples, -1),
                )
                qs_subset = tt.as_tensor_variable(np.flip(qs[:, :, :ind], axis=2))
                q_reg = tt.set_subtensor(
                    q_reg[:, :, i], tt.sum(tt.mul(qs_subset, ff_vec), axis=2)
                )
                pes_subset = tt.as_tensor_variable(np.flip(pes[:, :, :ind], axis=2))
                pe_reg = tt.set_subtensor(
                    pe_reg[:, :, i], tt.sum(tt.mul(pes_subset, ff_vec), axis=2),
                )
            pred_craving = w0_ + w1_ * q_reg - w2_ * pe_reg
            # pred_craving = tt.mul(w1, q_reg)
            pred = pm.Normal(
                "pred",
                mu=pred_craving,
                sigma=craving_sig,
                observed=norm_craving_ratings,
            )
            self.traces[pid] = pm.sample(
                draws=500,
                chains=2,
                cores=6,
                return_inferencedata=True,
                start={
                    "w0_mean": [0] * self.n_blocks,
                    "w1_mean": [0] * self.n_blocks,
                    "w2_mean": [0] * self.n_blocks,
                    "ff": 0.5,
                    "craving_sig": 0.5,
                },
                trace=[w0_mean, w1_mean, w2_mean, craving_sig, ff],
            )

        self.traces[pid].to_netcdf(
            f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
        )

    def fit_all(self, skip=None, jupyter=True):
        for i, pid in enumerate(self.longform["PID"].unique()):
            if jupyter:
                clear_output(wait=True)

            print(pid)
            print(f"Running participant {i+1} of {self.num_participants}")

            if pid in skip:
                continue
            # for i, pid in enumerate(
            #     ["58595b56a3149800011e156e", "615ddfac4254098aac0104f6"]
            # ):

            if os.path.exists(
                f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
            ):
                self.traces[pid] = az.from_netcdf(
                    f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
                )
            else:
                self.fit(pid)

    def load_completed(self, jupyter=True):
        for i, pid in enumerate(self.longform["PID"].unique()):

            if jupyter:
                clear_output(wait=True)

            print(pid)
            print(f"Running participant {i+1} of {self.num_participants}")
            if os.path.exists(
                f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
            ):
                print(f"Loading {pid}")
                self.traces[pid] = az.from_netcdf(
                    f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
                )
            else:
                print(f"{pid} not completed")

