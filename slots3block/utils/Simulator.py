import numpy as np
from model_functions.DecisionModels import Random, RescorlaWagner, RWDecay, FictiveRW, RWRP, EWA
from model_functions import PlottingPrototypes


class Simulator(object):
    """[summary]

    Args:
        object ([type]): [description]
    """

    def __init__(self, model_codes, num_trials, num_sims, reward_structure) -> None:
        super().__init__()
        self.model_dict = {}

        self.num_trials = num_trials
        self.num_sims = num_sims
        self.reward_structure = reward_structure

        if 'random' in model_codes:
            self.model_dict['random'] = Random.Random_SimActRew()
        if 'rw' in model_codes:
            self.model_dict['rw'] = RescorlaWagner.RW_SimActRew()
        if 'rwdecay' in model_codes:
            self.model_dict['rwdecay'] = RWDecay.RWDecay_SimActRew()
        if 'fictiverw' in model_codes:
            self.model_dict['fictiverw'] = FictiveRW.FictiveRW_SimActRew()
        if 'rwrp' in model_codes:
            self.model_dict['rwrp'] = RWRP.RWRP_SimActRew()
        if 'ewa' in model_codes:
            self.model_dict['ewa'] = EWA.EWA_SimActRew()

    def set_manual_params(self, model_code, param_set):
        """[summary]

        Args:
            model_code ([type]): [description]
            param_set ([type]): [description]
        """
        self.model_dict[model_code].choose_param_set(manual_param_set=param_set)

    def run_simulations(self):
        self.simulations = {}
        for model_code, model in self.model_dict.items():
            self.simulations[model_code] = {}
            actrew, param_set, values = model.simulate(self.reward_structure, self.num_trials, self.num_sims, set_high_beta=False)
            self.simulations[model_code]['actrew'] = actrew
            self.simulations[model_code]['param_set'] = param_set
            self.simulations[model_code]['values'] = values

    def plot_choices(self, model_code, adjust_size=(1000,1000)):
        return PlottingPrototypes.DecisionModelPlotting.plot_choices(self.simulations[model_code]['actrew'], adjust_size=adjust_size)

    def plot_choice_accuracy(self, model_code, reward_structure, adjust_size=(500,600)):
        PlottingPrototypes.DecisionModelPlotting.plot_choice_accuracy(self.simulations[model_code]['actrew'], reward_structure, adjust_size=adjust_size)

    @classmethod
    def get_model_codes(cls):
        return {
            'random': 'Random choices',
            'rw': 'Rescorla-Wagner',
            'rwdecay': 'RW with center decay',
            'fictiverw': 'RW with update of the unchosen option',
            'rwrp': 'RW with separate learning rates for +/- PEs',
            'ewa': 'Experience-weighted attraction'
        }
