from sepblock_craving.standard import EVRPE0step, EVRPE1step, EVRPE2step
from sepblock_craving.standard import EV0step, EV1step, EV2step
from sepblock_craving.standard import RPE0step, RPE1step, RPE2step
from sepblock_craving.standard import CueElic

# from sepblock_craving.delta import (
#     DeltaEVRPE0step,
#     DeltaEVRPE1step,
#     DeltaEVRPE2step,
# )
# from sepblock_craving.delta import DeltaRPE0step, DeltaRPE1step, DeltaRPE2step
# from sepblock_craving.delta import DeltaCueElic

from IPython.display import display, clear_output


class BatchFit(object):
    def __init__(
        self, chosen_models, longform, decision_model_path, craving_path, model_name
    ):
        self.longform = longform
        self.chosen_models = chosen_models
        self.craving_models = {}
        self.pid_subset = None

        for m in chosen_models:
            ## Standard models
            if m == "ev0step":
                self.craving_models[m] = EV0step.EV0step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            elif m == "ev1step":
                self.craving_models[m] = EV1step.EV1step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            elif m == "ev2step":
                self.craving_models[m] = EV2step.EV2step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            elif m == "rpe0step":
                self.craving_models[m] = RPE0step.RPE0step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            elif m == "rpe1step":
                self.craving_models[m] = RPE1step.RPE1step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            elif m == "rpe2step":
                self.craving_models[m] = RPE2step.RPE2step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            # elif m == "rpe2scue":
            #     self.craving_models[m] = RPE2sCue.RPE2sCue(
            #         longform,
            #         decision_model_path,
            #         craving_path,
            #         model_name,
            #         n_samples=2000,
            #     )
            elif m == "evrpe0step":
                self.craving_models[m] = EVRPE0step.EVRPE0step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            elif m == "evrpe1step":
                self.craving_models[m] = EVRPE1step.EVRPE1step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            elif m == "evrpe2step":
                self.craving_models[m] = EVRPE2step.EVRPE2step(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            elif m == "cueelic":
                self.craving_models[m] = CueElic.CueElic(
                    longform,
                    decision_model_path,
                    craving_path,
                    model_name,
                    n_samples=2000,
                )
            # ## Delta models
            # elif m == "deltarpe0step":
            #     self.craving_models[m] = DeltaRPE0step.DeltaRPE0step(
            #         longform,
            #         decision_model_path,
            #         craving_path,
            #         model_name,
            #         n_samples=2000,
            #     )
            # elif m == "deltarpe1step":
            #     self.craving_models[m] = DeltaRPE1step.DeltaRPE1step(
            #         longform,
            #         decision_model_path,
            #         craving_path,
            #         model_name,
            #         n_samples=2000,
            #     )
            # elif m == "deltarpe2step":
            #     self.craving_models[m] = DeltaRPE2step.DeltaRPE2step(
            #         longform,
            #         decision_model_path,
            #         craving_path,
            #         model_name,
            #         n_samples=2000,
            #     )
            # elif m == "deltaevrpe0step":
            #     self.craving_models[m] = DeltaEVRPE0step.DeltaEVRPE0step(
            #         longform,
            #         decision_model_path,
            #         craving_path,
            #         model_name,
            #         n_samples=2000,
            #     )
            # elif m == "deltaevrpe1step":
            #     self.craving_models[m] = DeltaEVRPE1step.DeltaEVRPE1step(
            #         longform,
            #         decision_model_path,
            #         craving_path,
            #         model_name,
            #         n_samples=2000,
            #     )
            # elif m == "deltaevrpe2step":
            #     self.craving_models[m] = DeltaEVRPE2step.DeltaEVRPE2step(
            #         longform,
            #         decision_model_path,
            #         craving_path,
            #         model_name,
            #         n_samples=2000,
            #     )
            # elif m == "deltacueelic":
            #     self.craving_models[m] = DeltaCueElic.DeltaCueElic(
            #         longform,
            #         decision_model_path,
            #         craving_path,
            #         model_name,
            #         n_samples=2000,
            #     )

    def load(self, jupyter=False):
        if jupyter:
            model_handle = display("Loading traces...", display_id=True)
        for m in self.chosen_models:
            self.craving_models[m].load_completed(jupyter=jupyter)
        if jupyter:
            clear_output(wait=True)

    def set_subset(self, model_subset=None, skip=None):
        self.model_subset = model_subset
        self.pid_subset = []
        for pid in self.craving_models[self.model_subset[0]].traces.keys():
            if pid not in self.longform["PID"].unique() or pid in skip:
                continue
            inall = True
            for model_name, model in self.craving_models.items():
                if model_subset is not None:
                    if model_name not in model_subset:
                        continue
                if pid not in model.ic["PID"].values:
                    inall = False
            if inall:
                self.pid_subset.append(pid)

        n_participants = len(self.pid_subset)
        print(f"{n_participants} participants in common in chosen models")

    def get_predictions(self, jupyter=False, rerun=False):
        if jupyter:
            model_handle = display("Calculating predictions...", display_id=True)
        for m in self.chosen_models:
            self.craving_models[m].all_sample_predictions(rerun=rerun, jupyter=jupyter)
        if jupyter:
            clear_output(wait=True)

    def get_ic(self, jupyter=False, rerun=False):
        if jupyter:
            model_handle = display("Calculating ICs...", display_id=True)
        for m in self.chosen_models:
            self.craving_models[m].all_sample_ic(rerun=rerun, jupyter=jupyter)
        if jupyter:
            clear_output(wait=True)

