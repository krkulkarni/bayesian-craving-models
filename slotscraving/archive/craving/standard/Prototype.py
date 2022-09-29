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
from IPython.display import display, clear_output


class Prototype(object):
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
        self.ll = {}
        self.predictions = {}
        self.norm_craving_ratings = {}
        self.norm_factors = {}

    def fit(self):
        raise NotImplementedError("This function is not implemented yet")

    # ImmEVRPE specific function to get sample predictions for a participant
    def get_sample_predictions(self):
        raise NotImplementedError("This function is not implemented yet")

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

        craving_inds = np.nonzero(craving_ratings[0] > -1)[0]
        craving_inds = np.delete(craving_inds, [12])
        craving_ratings = craving_ratings[:, craving_inds]

        ## NORMALIZE CRAVING RATINGS
        rating_means = np.reshape(np.repeat(np.mean(craving_ratings, axis=1), craving_ratings.shape[1]), (craving_ratings.shape))
        rating_stds = np.reshape(np.repeat(np.std(craving_ratings, axis=1), craving_ratings.shape[1]), (craving_ratings.shape))
        norm_factor = rating_stds[:,0]
        norm_craving_ratings = (craving_ratings - rating_means) / rating_stds
        norm_craving_ratings = np.tile(norm_craving_ratings, self.n_samples).reshape(self.n_blocks, self.n_samples, -1)

        ## RETRIEVE ALPHAS FROM DECISION MODEL
        traces = az.from_netcdf(f"{self.decision_path}/{self.model_name}/traces/{pid}.nc")
        alphas = np.vstack([traces.posterior.alpha.values[0], traces.posterior.alpha.values[1]])[: self.n_samples, :].T
        # fmt: on

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

        return (
            pid,
            actions,
            rewards,
            norm_craving_ratings,
            norm_factor,
            craving_inds,
            qs,
            pes,
        )

    ## Generic function to fit all participants
    def fit_all(self, cores=4, draws=1000, skip=None):
        if skip is None:
            skip = []

        for i, pid in enumerate(self.longform["PID"].unique()):

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
        if jupyter:
            pid_handle = display("pid", display_id=True)
        for i, pid in enumerate(self.longform["PID"].unique()):
            if jupyter:
                pid_handle.update(
                    f"{self.craving_model_name} | Participant {i+1} of {self.num_participants}: {pid}"
                )

            # print(f"Running participant {i+1} of {self.num_participants}")
            if pid not in self.traces.keys():
                if os.path.exists(
                    f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
                ):
                    self.traces[pid] = az.from_netcdf(
                        f"{self.craving_path}/{self.craving_model_name}/traces/{pid}.nc"
                    )

    # Generic function to calculate BIC and DIC for a given participant from true ratings and predictions
    # Uses cdf to calculate probability of true rating (e.g. if true rating is 22, then calculate probability between 21.5 and 22.5)
    # In the normalized ratings, the range is -1 to 1 instead of 0 to 50, so offset translates to +/- 0.2
    def calculate_ic(self, pid, sample_preds, norm_craving_ratings, norm_factor):

        # offset = 0.2

        if not os.path.exists(f"{self.craving_path}/{self.craving_model_name}/loglik/"):
            os.makedirs(f"{self.craving_path}/{self.craving_model_name}/loglik/")

        craving_sig_samples = np.vstack(
            [
                self.traces[pid].posterior.craving_sig.values[0, :, :],
                self.traces[pid].posterior.craving_sig.values[1, :, :],
            ]
        )
        mean_craving_sig = np.mean(craving_sig_samples, axis=0)

        # fmt: off
        if os.path.exists(f"{self.craving_path}/{self.craving_model_name}/loglik/{pid}.csv"):
            with open(f"{self.craving_path}/{self.craving_model_name}/loglik/{pid}.csv", "r") as f:
                self.ll[pid] = np.loadtxt(f, delimiter=",")

        else:
            self.ll[pid] = np.zeros((self.n_blocks, self.n_samples))
            for b, block in enumerate(["money", "other"]):
                block_nf = norm_factor[b]
                offset = (1/block_nf)/2
                for samp in np.arange(self.n_samples):
                    acc = 0
                    for p, t in zip(sample_preds[samp][b], norm_craving_ratings[b, samp, :]):
                        acc += np.log(
                            norm.cdf(t + offset, p, craving_sig_samples[samp, b])
                            - norm.cdf(t - offset, p, craving_sig_samples[samp, b])
                        )
                    self.ll[pid][b, samp] = acc
        # fmt: on

        bic = self.n_params * np.log(20) - 2 * self.ll[pid]
        weighted_mean_bic = bic.mean(axis=1) * np.square(mean_craving_sig)

        pdic = np.array(
            [
                2 * (self.ll[pid][:, elem] - self.ll[pid].mean(axis=1))
                for elem in np.arange(self.n_samples)
            ]
        ).T
        dic = -2 * self.ll[pid] + 2 * pdic
        weighted_mean_dic = dic.mean(axis=1) * np.square(mean_craving_sig)

        np.savetxt(
            f"{self.craving_path}/{self.craving_model_name}/loglik/{pid}.csv",
            self.ll[pid],
            delimiter=",",
        )

        return bic, weighted_mean_bic, dic, weighted_mean_dic

    # Generic function to calculate the participant predictions
    # DEPENDS ON RW-SPECIFIC _get_rw_actrewrate_qspes function
    def all_sample_predictions(self, rerun=False, jupyter=False):

        # fmt: off
        if jupyter:
            pid_handle = display(
                f"{self.craving_model_name} | Loading from stored predictions",
                display_id=True,
            )
        for i, pid in enumerate(self.traces.keys()):
            if jupyter:
                pid_handle.update(f"{self.craving_model_name} | Participant {i+1} of {len(self.traces.keys())}: {pid}")
            if pid not in self.predictions.keys() or rerun:
                if (
                    not rerun
                    and os.path.exists(f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}.npy")
                    and os.path.exists(f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}.npy")
                ):
                    self.predictions[pid] = np.load(f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}.npy")
                    self.norm_craving_ratings[pid] = np.load(f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}.npy")

                else:
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

                    self.norm_factors[pid] = norm_factor
                    self.predictions[pid] = self.get_sample_predictions(pid, actions, rewards, norm_craving_ratings, craving_inds, qs, pes)
                    np.save(
                        f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}.npy",
                        self.predictions[pid],
                    )
                    self.norm_craving_ratings[pid] = norm_craving_ratings
                    np.savetxt(
                        f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}.csv",
                        self.norm_craving_ratings[pid][:, 0, :],
                        delimiter=",",
                    )
        # fmt: on

    # Generic function to calculate the BIC and DIC for all participants
    def all_sample_ic(self, rerun=False, jupyter=False):

        if not (hasattr(self, "predictions") and hasattr(self, "norm_craving_ratings")):
            raise ValueError("Run all_sample_predictions first")

        # fmt: off
        if not rerun and os.path.exists(f"{self.craving_path}/{self.craving_model_name}/ic.csv"):
            self.ic = pd.read_csv(f"{self.craving_path}/{self.craving_model_name}/ic.csv")
        else:
            self.ic = pd.DataFrame(columns=["PID", "Money BIC", "Other BIC", "Money DIC", "Other DIC"])
        # fmt: on

        if jupyter:
            pid_handle = display(
                f"{self.craving_model_name} | Loading from stored IC", display_id=True
            )
        for i, pid in enumerate(self.traces.keys()):
            if jupyter:
                pid_handle.update(
                    f"{self.craving_model_name} | Participant {i+1} of {len(self.traces.keys())}: {pid}"
                )
            if pid in self.ic["PID"].unique():
                continue

            _, bic, _, dic = self.calculate_ic(
                pid,
                self.predictions[pid],
                self.norm_craving_ratings[pid],
                self.norm_factors[pid],
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
            )
        # fmt: off
        self.ic.to_csv(f"{self.craving_path}/{self.craving_model_name}/ic.csv", index=False)
        # fmt: on

        return self.ic
