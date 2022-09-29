# import numpy as np
from pyexpat import model
from slots3block.decision import (
    Random,
    RescorlaWagner,
    RWDecay,
    FictiveRW,
    RWRP,
    EWA,
    Heuristic,
)
import importlib

importlib.reload(Random)
importlib.reload(RescorlaWagner)
importlib.reload(RWDecay)
importlib.reload(FictiveRW)
importlib.reload(RWRP)
importlib.reload(EWA)
from slots3block.utils import PlottingPrototypes


class DecisionBatchFit(object):
    def __init__(self, model_codes, concat=False) -> None:
        super().__init__()
        self.model_dict = {}
        self.concat = concat

        self.model_codes = model_codes

        if "random" in model_codes:
            self.model_dict["random"] = Random.Random()
        if "nwsls" in model_codes:
            self.model_dict["nwsls"] = Heuristic.NWSLS()
        if "rw" in model_codes:
            self.model_dict["rw"] = RescorlaWagner.RW()
        if "rwdecay" in model_codes:
            self.model_dict["rwdecay"] = RWDecay.RWDecay()
        if "fictiverw" in model_codes:
            self.model_dict["fictiverw"] = FictiveRW.FictiveRW()
        if "rwrp" in model_codes:
            self.model_dict["rwrp"] = RWRP.RWRP()
        if "ewa" in model_codes:
            self.model_dict["ewa"] = EWA.EWA()

    def fit_all(self, actrew, from_file=None):
        if from_file:
            # load from file...
            # return structures
            pass

        for model_name, model in self.model_dict.items():
            if model_name == "random" or model_name == "nwsls":
                _ = model.fit_all(
                    actrew, concat=self.concat, skip_prediction_error=True
                )
            else:
                _ = model.fit_all(actrew, self.concat)

    def model_comparison(self, sums=False, streamlit=False, adjust_size=(700, 1000)):
        all_fit_metrics = [
            fit_model.fit_metrics for fit_model in self.model_dict.values()
        ]
        model_names = list(self.model_dict.keys())
        return PlottingPrototypes.DecisionModelPlotting.model_comparison(
            all_fit_metrics, model_names, sums=sums, adjust_size=adjust_size, streamlit=streamlit
        )

    def plot_parameters(
        self,
        model_code,
        chosen_params=None,
        true_params=None,
        adjust_size=(500, 800),
        update_to_range=None,
    ):
        PlottingPrototypes.DecisionModelPlotting.plot_parameters(
            self.model_dict[model_code].fit_metrics,
            chosen_params=chosen_params,
            true_params=true_params,
            adjust_size=adjust_size,
            update_to_range=update_to_range,
        )

    def plot_values(self, adjust_size=(700, 800), update_to_range=[-2, 4]):
        all_fit_metrics = [
            fit_model.fit_metrics for fit_model in self.model_dict.values()
        ]
        model_names = list(self.model_dict.keys())
        PlottingPrototypes.DecisionModelPlotting.plot_values(
            all_fit_metrics,
            model_names,
            adjust_size=adjust_size,
            update_to_range=update_to_range,
        )


# class CravingBatchFit(object):

#     def __init__(self, model_codes) -> None:
#         super().__init__()
#         self.model_dict = {}

#         self.model_codes = model_codes

#         if 'random' in model_codes:
#             self.model_dict['random'] = Random.Random()
#         if 'rw' in model_codes:
#             self.model_dict['rw'] = RescorlaWagner.RW()
#         if 'rwdecay' in model_codes:
#             self.model_dict['rwdecay'] = RWDecay.RWDecay()
#         if 'fictiverw' in model_codes:
#             self.model_dict['fictiverw'] = FictiveRW.FictiveRW()
#         if 'rwrp' in model_codes:
#             self.model_dict['rwrp'] = RWRP.RWRP()
#         if 'ewa' in model_codes:
#             self.model_dict['ewa'] = EWA.EWA()

#     def fit_all(self, actrew, from_file=None):
#         if from_file:
#             # load from file...
#             # return structures
#             pass

#         for model_name, model in self.model_dict.items():
#             if model_name == 'random':
#                 _ = model.fit_all(actrew, skip_prediction_error=True)
#             else:
#                 _ = model.fit_all(actrew)

#     def model_comparison(self, sums=False, adjust_size=(700,1000)):
#         all_fit_metrics = [fit_model.fit_metrics for fit_model in self.model_dict.values()]
#         model_names = list(self.model_dict.keys())
#         PlottingPrototypes.DecisionModelPlotting.model_comparison(all_fit_metrics, model_names, sums=sums, adjust_size=adjust_size)

#     def plot_parameters(self, model_code, true_params=None, adjust_size=(500,800)):
#         PlottingPrototypes.DecisionModelPlotting.plot_parameters(self.model_dict[model_code].fit_metrics, true_params=true_params, adjust_size=adjust_size)

