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


class DeltaEVRPE0step(object):
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
        self.craving_model_name = "DeltaEVRPE0step"
        self.equation = "w0 + w1 * Q - w2 * PE"
        self.n_params = 3
        ##--------------------------------##
        ###############################################################################

        ##--------- DO NOT ALTER ---------##
        self.longform = longform
        self.num_participants = len(longform["PID"].unique())

        self.n_blocks = n_blocks
        self.n_samples = n_samples
        self.n_trials = n_trials

        self.decision_path = decision_path
        self.model_name = model_name

        # fmt: off
        self.craving_path = craving_path
        if not os.path.exists(f"{self.craving_path}/{self.craving_model_name}/"):
            os.makedirs(f"{self.craving_path}/{self.craving_model_name}/")
        if not os.path.exists(f"{self.craving_path}/{self.craving_model_name}/traces/"):
            os.makedirs(f"{self.craving_path}/{self.craving_model_name}/traces/")
        if not os.path.exists(f"{self.craving_path}/{self.craving_model_name}/predictions/"):
            os.makedirs(f"{self.craving_path}/{self.craving_model_name}/predictions/")
        if not os.path.exists(f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/"):
            os.makedirs(f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/")
        # fmt: on

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

    def fit(self, pid, draws=1000, cores=4):

        (
            pid,
            actions,
            rewards,
            norm_craving_ratings,
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
            
            craving_sig = pm.Exponential('craving_sig', lam=1)

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
                sigma=craving_sig,
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

    ## RW MODEL SPECIFIC Q CALCULATION ##
    def _get_rw_actrewrate_qspes(self, pid):
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
        delta_craving_ratings = np.zeros(craving_ratings.shape)
        for b, block in enumerate(['money', 'other']):
            delta_craving_ratings[b, :] = np.hstack([[0], np.diff(craving_ratings[b, :])])
        # fmt: on

        ## NORMALIZE CRAVING RATINGS
        craving_inds = np.nonzero(craving_ratings[0] > -1)[0]
        craving_inds = np.delete(craving_inds, [12])

        delta_craving_ratings = np.tile(
            delta_craving_ratings[:, craving_inds], self.n_samples
        ).reshape(self.n_blocks, self.n_samples, -1)
        norm_craving_ratings = delta_craving_ratings / 50
        # norm_craving_ratings = (craving_ratings - 25) / 25

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

        return pid, actions, rewards, norm_craving_ratings, craving_inds, qs, pes

    ## Generic function to fit all participants
    def fit_all(self, cores=4, draws=1000, skip=None, jupyter=True):
        if skip is None:
            skip = []
        for i, pid in enumerate(self.longform["PID"].unique()):
            if jupyter:
                clear_output(wait=True)

            print(pid)
            print(f"Running participant {i+1} of {self.num_participants}")

            if pid in skip:
                continue

            if os.path.exists(
                f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
            ):
                self.traces[pid] = az.from_netcdf(
                    f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
                )
            else:
                self.fit(pid, cores=cores, draws=draws)

    # Generic function to load all completed participants
    def load_completed(self, jupyter=True):
        for i, pid in enumerate(self.longform["PID"].unique()):

            if jupyter:
                clear_output(wait=True)

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

    # Generic function to calculate BIC and DIC for a given participant from true ratings and predictions
    def calculate_ic(self, pid, sample_preds, norm_craving_ratings):

        craving_sig_samples = np.hstack(
            [
                self.traces[pid].posterior.craving_sig.values[0, :],
                self.traces[pid].posterior.craving_sig.values[1, :],
            ]
        )

        ll = np.zeros((self.n_blocks, self.n_samples))
        for b, block in enumerate(["money", "other"]):
            for samp in np.arange(self.n_samples):
                acc = 0
                for p, t in zip(
                    sample_preds[samp][b], norm_craving_ratings[b, samp, :]
                ):
                    acc += np.log(norm.pdf(t, p, craving_sig_samples[samp]))
                ll[b, samp] = acc

        bic = self.n_params * np.log(20) - 2 * ll
        mean_bic = bic.mean(axis=1)

        pdic = np.array(
            [2 * (ll[:, elem] - ll.mean(axis=1)) for elem in np.arange(self.n_samples)]
        ).T
        dic = -2 * ll + 2 * pdic
        mean_dic = dic.mean(axis=1)

        return bic, mean_bic, dic, mean_dic

    # Generic function to calculate the participant predictions
    # DEPENDS ON RW-SPECIFIC _get_rw_actrewrate_qspes function
    def all_sample_predictions(self, rerun=False, jupyter=True):

        self.predictions = {}
        self.norm_craving_ratings = {}

        for i, pid in enumerate(self.traces.keys()):

            if jupyter:
                clear_output(wait=True)

            print(f"Calculating participant {i+1} of {len(self.traces.keys())}")

            if (
                not rerun
                and os.path.exists(
                    f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}.npy"
                )
                and os.path.exists(
                    f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}.npy"
                )
            ):
                self.predictions[pid] = np.load(
                    f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}.npy"
                )
                self.norm_craving_ratings[pid] = np.load(
                    f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}.npy"
                )

            else:
                (
                    pid,
                    actions,
                    rewards,
                    norm_craving_ratings,
                    craving_inds,
                    qs,
                    pes,
                ) = self._get_rw_actrewrate_qspes(pid)

                self.predictions[pid] = self.get_sample_predictions(
                    pid, actions, rewards, norm_craving_ratings, craving_inds, qs, pes
                )
                np.save(
                    f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}.npy",
                    self.predictions[pid],
                )
                self.norm_craving_ratings[pid] = norm_craving_ratings
                np.save(
                    f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}.npy",
                    self.norm_craving_ratings[pid],
                )

    # Generic function to calculate the BIC and DIC for all participants
    def all_sample_ic(self, rerun=False, jupyter=True):

        if not (hasattr(self, "predictions") and hasattr(self, "norm_craving_ratings")):
            raise ValueError("Run all_sample_predictions first")

        if not rerun and os.path.exists(
            f"{self.craving_path}/{self.craving_model_name}/ic.csv"
        ):
            self.ic = pd.read_csv(
                f"{self.craving_path}/{self.craving_model_name}/ic.csv"
            )
            for i, pid in enumerate(self.predictions.keys()):
                if pid in self.ic["PID"].unique():
                    continue

                if jupyter:
                    clear_output(wait=True)

                print(
                    f"Calculating participant {i+1} of {len(self.predictions.keys())}"
                )

                _, bic, _, dic = self.calculate_ic(
                    pid, self.predictions[pid], self.norm_craving_ratings[pid]
                )

                self.ic = pd.concat(
                    [
                        self.ic,
                        pd.DataFrame(
                            {
                                "PID": pid,
                                "Money BIC": bic[0],
                                "Other BIC": bic[1],
                                "Money DIC": dic[0],
                                "Other DIC": dic[1],
                            },
                            index=[i],
                        ),
                    ]
                )  # .reset_index()
            self.ic.to_csv(
                f"{self.craving_path}/{self.craving_model_name}/ic.csv", index=False
            )

        else:
            self.ic = pd.DataFrame()
            for i, pid in enumerate(self.predictions.keys()):

                if jupyter:
                    clear_output(wait=True)

                print(
                    f"Calculating participant {i+1} of {len(self.predictions.keys())}"
                )

                _, bic, _, dic = self.calculate_ic(
                    pid, self.predictions[pid], self.norm_craving_ratings[pid]
                )

                self.ic = pd.concat(
                    [
                        self.ic,
                        pd.DataFrame(
                            {
                                "PID": pid,
                                "Money BIC": bic[0],
                                "Other BIC": bic[1],
                                "Money DIC": dic[0],
                                "Other DIC": dic[1],
                            },
                            index=[i],
                        ),
                    ]
                )  # .reset_index()
            self.ic.to_csv(
                f"{self.craving_path}/{self.craving_model_name}/ic.csv", index=False
            )

        return self.ic
