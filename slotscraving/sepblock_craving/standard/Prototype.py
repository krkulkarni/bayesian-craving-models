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
from scipy.special import expit
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
        self.skipped_list = []

    def _fit(self):
        raise NotImplementedError("This function is not implemented yet")

    # ImmEVRPE specific function to get sample predictions for a participant
    def get_sample_predictions(self):
        raise NotImplementedError("This function is not implemented yet")

    def _normalize_ratings(self, craving_ratings):

        rating_mean = np.mean(craving_ratings)
        rating_std = np.std(craving_ratings)
        norm_factor = rating_std
        if not norm_factor == 0:
            norm_craving_ratings = (craving_ratings - rating_mean) / rating_std
        else:
            norm_craving_ratings = craving_ratings
            print("No variation in craving ratings")
            print(craving_ratings)

        norm_craving_ratings = np.tile(norm_craving_ratings, self.n_samples).reshape(
            self.n_samples, -1
        )

        return norm_craving_ratings, norm_factor

    ## RW MODEL SPECIFIC Q CALCULATION ##
    def _get_rw_actrewrate_qspes(self, pid, block, norm_only=False):
        ## RETRIEVE ACTIONS, REWARDS, CRAVING RATINGS
        # fmt: off
        actions = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Action'].values.astype(int)
        rewards = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Reward'].values.astype(int)
        craving_ratings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values

        craving_inds = np.nonzero(craving_ratings > -1)
        craving_inds = np.delete(craving_inds, [12])
        craving_ratings = craving_ratings[craving_inds]

        ## NORMALIZE CRAVING RATINGS
        # rating_means = np.reshape(np.repeat(np.mean(craving_ratings, axis=1), craving_ratings.shape[1]), (craving_ratings.shape))
        # rating_stds = np.reshape(np.repeat(np.std(craving_ratings, axis=1), craving_ratings.shape[1]), (craving_ratings.shape))
        # norm_factor = rating_stds[:,0]
        # norm_craving_ratings = (craving_ratings - rating_means) / rating_stds
        # norm_craving_ratings = np.tile(norm_craving_ratings, self.n_samples).reshape(self.n_blocks, self.n_samples, -1)
        norm_craving_ratings, norm_factor = self._normalize_ratings(craving_ratings)
        if norm_only:
            return norm_factor

        ## RETRIEVE ALPHAS FROM DECISION MODEL
        try:
            traces = self.traces[pid][block]
        except KeyError:
            print('Loading traces from file')
            traces = az.from_netcdf(f"{self.decision_path}/{self.model_name}/traces/{pid}_{block}.nc")
        alphas = np.hstack([traces.posterior.alpha.values[0], traces.posterior.alpha.values[1]])[:self.n_samples].T
        # fmt: on

        ## GENERATE QS AND PES
        qs = np.zeros((self.n_samples, self.n_trials))
        pes = np.zeros((self.n_samples, self.n_trials))

        for i in np.arange(self.n_samples):
            al = alphas[i]
            qs_ = np.array([0.5, 0.5])
            chosen_qs = [0.5]
            chosen_pes = []
            for a, r in zip(actions, rewards):
                pe = r - qs_[a]
                chosen_pes.append(pe)
                qs_[a] = qs_[a] + al * pe
                chosen_qs.append(qs_[a])
            qs[i, :] = chosen_qs[:-1]
            pes[i, :] = chosen_pes

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

    def _get_rwdecay_actrewrate_qspes(self, pid, block, norm_only=False):
        ## RETRIEVE ACTIONS, REWARDS, CRAVING RATINGS
        # fmt: off
        actions = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Action'].values.astype(int)
        rewards = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Reward'].values.astype(int)
        craving_ratings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values

        craving_inds = np.nonzero(craving_ratings > -1)
        craving_inds = np.delete(craving_inds, [12])
        craving_ratings = craving_ratings[craving_inds]

        ## NORMALIZE CRAVING RATINGS
        # rating_means = np.reshape(np.repeat(np.mean(craving_ratings, axis=1), craving_ratings.shape[1]), (craving_ratings.shape))
        # rating_stds = np.reshape(np.repeat(np.std(craving_ratings, axis=1), craving_ratings.shape[1]), (craving_ratings.shape))
        # norm_factor = rating_stds[:,0]
        # norm_craving_ratings = (craving_ratings - rating_means) / rating_stds
        # norm_craving_ratings = np.tile(norm_craving_ratings, self.n_samples).reshape(self.n_blocks, self.n_samples, -1)
        norm_craving_ratings, norm_factor = self._normalize_ratings(craving_ratings)
        if norm_only:
            return norm_factor

        ## RETRIEVE ALPHAS FROM DECISION MODEL
        try:
            traces = self.traces[pid][block]
        except KeyError:
            print('Loading traces from file')
            traces = az.from_netcdf(f"{self.decision_path}/{self.model_name}/traces/{pid}_{block}.nc")
        alphas = np.hstack([traces.posterior.alpha.values[0], traces.posterior.alpha.values[1]])[:self.n_samples].T
        decays = np.hstack([traces.posterior.decay.values[0], traces.posterior.decay.values[1]])[:self.n_samples].T
        # fmt: on

        ## GENERATE QS AND PES
        qs = np.zeros((self.n_samples, self.n_trials))
        pes = np.zeros((self.n_samples, self.n_trials))

        for i in np.arange(self.n_samples):
            al = alphas[i]
            d = decays[i]
            qs_ = np.array([0.5, 0.5])
            chosen_qs = [0.5]
            chosen_pes = []
            for a, r in zip(actions, rewards):
                pe = r - qs_[a]
                chosen_pes.append(pe)
                qs_[a] = qs_[a] + al * pe
                qs_[1-a] = qs_[1-a] * (1-expit(d)) + expit(d) * 0.5
                chosen_qs.append(qs_[a])
            qs[i, :] = chosen_qs[:-1]
            pes[i, :] = chosen_pes

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

    def _get_rwrl_actrewrate_qspes(self, pid, block, norm_only=False):
        ## RETRIEVE ACTIONS, REWARDS, CRAVING RATINGS
        # fmt: off
        actions = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Action'].values.astype(int)
        rewards = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Reward'].values.astype(int)
        craving_ratings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values

        craving_inds = np.nonzero(craving_ratings > -1)
        craving_inds = np.delete(craving_inds, [12])
        craving_ratings = craving_ratings[craving_inds]

        ## NORMALIZE CRAVING RATINGS
        # rating_means = np.reshape(np.repeat(np.mean(craving_ratings, axis=1), craving_ratings.shape[1]), (craving_ratings.shape))
        # rating_stds = np.reshape(np.repeat(np.std(craving_ratings, axis=1), craving_ratings.shape[1]), (craving_ratings.shape))
        # norm_factor = rating_stds[:,0]
        # norm_craving_ratings = (craving_ratings - rating_means) / rating_stds
        # norm_craving_ratings = np.tile(norm_craving_ratings, self.n_samples).reshape(self.n_blocks, self.n_samples, -1)
        norm_craving_ratings, norm_factor = self._normalize_ratings(craving_ratings)
        if norm_only:
            return norm_factor

        ## RETRIEVE ALPHAS FROM DECISION MODEL
        try:
            traces = self.traces[pid][block]
        except KeyError:
            print('Loading traces from file')
            traces = az.from_netcdf(f"{self.decision_path}/{self.model_name}/traces/{pid}_{block}.nc")
        alphas_pos = np.hstack([traces.posterior.alpha_pos.values[0], traces.posterior.alpha_pos.values[1]])[:self.n_samples].T
        alphas_neg = np.hstack([traces.posterior.alpha_neg.values[0], traces.posterior.alpha_neg.values[1]])[:self.n_samples].T
        # fmt: on

        ## GENERATE QS AND PES
        qs = np.zeros((self.n_samples, self.n_trials))
        pes = np.zeros((self.n_samples, self.n_trials))

        for i in np.arange(self.n_samples):
            pos_al = alphas_pos[i]
            neg_al = alphas_neg[i]
            qs_ = np.array([0.5, 0.5])
            chosen_qs = [0.5]
            chosen_pes = []
            for a, r in zip(actions, rewards):
                pe = r - qs_[a]
                chosen_pes.append(pe)
                if pe >=0:
                    qs_[a] = qs_[a] + pos_al * pe
                else:
                    qs_[a] = qs_[a] + neg_al * pe
                chosen_qs.append(qs_[a])
            qs[i, :] = chosen_qs[:-1]
            pes[i, :] = chosen_pes

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
    def fit_all(self, select_pids=None, cores=4, draws=2000, skip=None):
        if skip is None:
            skip = []
        if select_pids is None:
            select_pids = self.longform["PID"].unique()

        for i, pid in enumerate(select_pids):
            print(f"Running participant {i+1} of {self.num_participants}")
            if pid in skip:
                print(f"Skipping participant {i+1}: {pid}")
                continue
            else:
                self._fit(pid, cores=cores, draws=draws)

    # Generic function to load all completed participants
    def load_completed(self, jupyter=True):
        if jupyter:
            pid_handle = display("pid", display_id=True)
        for i, pid in enumerate(self.longform["PID"].unique()):
            if jupyter:
                pid_handle.update(
                    f"{self.craving_model_name} | Participant {i+1} of {self.num_participants}: {pid}"
                )
            for block in ["money", "other"]:
                if os.path.exists(
                    f"{self.craving_path}/{self.craving_model_name}/traces/{pid}_{block}.nc"
                ):
                    if pid not in self.traces.keys():
                        self.traces[pid] = {}
                    self.traces[pid][block] = az.from_netcdf(
                        f"{self.craving_path}/{self.craving_model_name}/traces/{pid}_{block}.nc"
                    )
                else:
                    # self.traces[pid][block] = "RuntimeError or other issue"
                    self.skipped_list.append([f'{pid}_{block}'])

    # Generic function to calculate BIC and DIC for a given participant from true ratings and predictions
    # Uses cdf to calculate probability of true rating (e.g. if true rating is 22, then calculate probability between 21.5 and 22.5)
    # In the normalized ratings, the range is -1 to 1 instead of 0 to 50, so offset translates to +/- 0.2
    def calculate_ic(
        self, pid, block, n_samples, sample_preds, norm_craving_ratings, norm_factor
    ):

        if pid not in self.ll.keys():
            self.ll[pid] = {}
        if block not in self.ll[pid].keys():
            self.ll[pid][block] = {}

        if not os.path.exists(f"{self.craving_path}/{self.craving_model_name}/loglik/"):
            os.makedirs(f"{self.craving_path}/{self.craving_model_name}/loglik/")

        try:
            craving_sig_samples = np.hstack(
                [
                    self.traces[pid][block].posterior.craving_sig.values[0, :],
                    self.traces[pid][block].posterior.craving_sig.values[1, :],
                ]
            )
            mean_craving_sig = np.mean(craving_sig_samples)

            # fmt: off
            if os.path.exists(f"{self.craving_path}/{self.craving_model_name}/loglik/{pid}_{block}.csv"):
                with open(f"{self.craving_path}/{self.craving_model_name}/loglik/{pid}_{block}.csv", "r") as f:
                    self.ll[pid][block] = np.loadtxt(f, delimiter=",")

            else:
                self.ll[pid][block] = np.zeros(n_samples)
                offset = (1/norm_factor)/2
                for samp in np.arange(n_samples):
                    acc = 0
                    for p, t in zip(sample_preds[samp,:], norm_craving_ratings[samp, :]):
                        acc += np.log(
                            norm.cdf(t + offset, p, craving_sig_samples[samp])
                            - norm.cdf(t - offset, p, craving_sig_samples[samp])
                        )
                    self.ll[pid][block][samp] = acc
            # fmt: on

            bic = self.n_params * np.log(20) - 2 * self.ll[pid][block]
            weighted_mean_bic = bic.mean() * np.square(mean_craving_sig)

            pdic = np.array(
                [
                    2 * (self.ll[pid][block][elem] - self.ll[pid][block].mean())
                    for elem in np.arange(n_samples)
                ]
            ).T
            dic = -2 * self.ll[pid][block] + 2 * pdic
            weighted_mean_dic = dic.mean() * np.square(mean_craving_sig)

            np.savetxt(
                f"{self.craving_path}/{self.craving_model_name}/loglik/{pid}_{block}.csv",
                self.ll[pid][block],
                delimiter=",",
            )

            return bic, weighted_mean_bic, dic, weighted_mean_dic
        except AttributeError:
            print(f"Posterior not calculated for {pid}_{block}")
            return -1, -1, -1, -1

    # Generic function to calculate the participant predictions
    # DEPENDS ON RW-SPECIFIC _get_rw_actrewrate_qspes function
    def all_sample_predictions(self, select_pids=None, rerun=False, jupyter=False):

        # fmt: off
        if jupyter:
            pid_handle = display(
                f"{self.craving_model_name} | Loading from stored predictions",
                display_id=True,
            )
        if select_pids is None:
            select_pids = self.traces.keys()
            
        for i, pid in enumerate(select_pids):
            if jupyter:
                pid_handle.update(f"{self.craving_model_name} | Participant {i+1} of {len(self.traces.keys())}: {pid}")
            if pid not in self.predictions.keys() or rerun:
                self.predictions[pid] = {}
                self.norm_craving_ratings[pid] = {}
                self.norm_factors[pid] = {}
                for block in ["money", "other"]:

                    if (
                        not rerun
                        and os.path.exists(f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}_{block}.npy")
                        and os.path.exists(f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}_{block}.csv")
                    ):
                        self.norm_factors[pid][block] = self._get_rwrl_actrewrate_qspes(pid, block, norm_only=True)
                        self.predictions[pid][block] = np.load(f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}_{block}.npy", allow_pickle=True)
                        self.norm_craving_ratings[pid][block] = np.loadtxt(f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}_{block}.csv")

                    else:
                        if self.model_name=='rwrl' or self.model_name=='RWRL':
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
                        elif self.model_name=='rw' or self.model_name=='RW':
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
                        else:
                            raise ValueError(f"{self.model_name} is not a valid model name")

                        self.norm_factors[pid][block] = norm_factor
                        self.predictions[pid][block] = self.get_sample_predictions(pid, block, actions, rewards, norm_craving_ratings, craving_inds, qs, pes)
                        np.save(
                            f"{self.craving_path}/{self.craving_model_name}/predictions/{pid}_{block}.npy",
                            self.predictions[pid][block],
                        )
                        self.norm_craving_ratings[pid][block] = norm_craving_ratings
                        np.savetxt(
                            f"{self.craving_path}/{self.craving_model_name}/norm_craving_ratings/{pid}_{block}.csv",
                            self.norm_craving_ratings[pid][block][0, :],
                            delimiter=",",
                        )
        # fmt: on

    # Generic function to calculate the BIC and DIC for all participants
    def all_sample_ic(self, n_samples=None, rerun=False, jupyter=False):

        if not (hasattr(self, "predictions") and hasattr(self, "norm_craving_ratings")):
            raise ValueError("Run all_sample_predictions first")

        if n_samples is None:
            n_samples = self.n_samples

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

            block_bics, block_dics = (
                np.zeros(self.n_blocks) - 1,
                np.zeros(self.n_blocks) - 1,
            )
            for b, block in enumerate(["money", "other"]):

                _, block_bics[b], _, block_dics[b] = self.calculate_ic(
                    pid,
                    block,
                    n_samples,
                    self.predictions[pid][block],
                    self.norm_craving_ratings[pid][block],
                    self.norm_factors[pid][block],
                )

            self.ic = pd.concat(
                [
                    self.ic,
                    pd.DataFrame(
                        {
                            "PID": pid,
                            "Money BIC": block_bics[0],
                            "Other BIC": block_bics[1],
                            "Money DIC": block_dics[0],
                            "Other DIC": block_dics[1],
                        },
                        index=[i],
                    ),
                ]
            )
        # fmt: off
        self.ic.replace(-1, np.nan, inplace=True)
        self.ic.to_csv(f"{self.craving_path}/{self.craving_model_name}/ic.csv", index=False)
        # fmt: on

        return self.ic
