## Latest version of the models for the slotscraving task
## Author: Kaustubh Kulkarni
## Original Date: June 2022
## Last Modified: Feb 5, 2022

from IPython.display import clear_output

from .cm_models.active_cm_models import A_RW_0step, A_RW_1stepDecay, A_RW_1stepMean, A_RW_1stepSep, A_RW_2stepDecay, A_RW_2stepMean, A_RW_2stepSep, A_RWSep_0step, A_RWSep_1stepDecay, A_RWSep_1stepMean, A_RWSep_1stepSep, A_RWSep_2stepDecay, A_RWSep_2stepMean, A_RWSep_2stepSep

from .cm_models.active_beta_cm_models import A_Beta_RW_0step, A_Beta_RW_1stepDecay, A_Beta_RW_1stepMean, A_Beta_RW_1stepSep, A_Beta_RW_2stepDecay, A_Beta_RW_2stepMean, A_Beta_RW_2stepSep, A_Beta_RWSep_0step, A_Beta_RWSep_1stepDecay, A_Beta_RWSep_1stepMean, A_Beta_RWSep_1stepSep, A_Beta_RWSep_2stepDecay, A_Beta_RWSep_2stepMean, A_Beta_RWSep_2stepSep

from .cm_models.active_mult_cm_models import A_Mult_RW_0step, A_Mult_RW_1stepDecay, A_Mult_RW_1stepMean, A_Mult_RW_1stepSep, A_Mult_RW_2stepDecay, A_Mult_RW_2stepMean, A_Mult_RW_2stepSep, A_Mult_RWSep_0step, A_Mult_RWSep_1stepDecay, A_Mult_RWSep_1stepMean, A_Mult_RWSep_1stepSep, A_Mult_RWSep_2stepDecay, A_Mult_RWSep_2stepMean, A_Mult_RWSep_2stepSep

from .cm_models.active_rew_cm_models import A_Rew_RW_0step, A_Rew_RW_1stepDecay, A_Rew_RW_1stepMean, A_Rew_RW_1stepSep, A_Rew_RW_2stepDecay, A_Rew_RW_2stepMean, A_Rew_RW_2stepSep, A_Rew_RWSep_0step, A_Rew_RWSep_1stepDecay, A_Rew_RWSep_1stepMean, A_Rew_RWSep_1stepSep, A_Rew_RWSep_2stepDecay, A_Rew_RWSep_2stepMean, A_Rew_RWSep_2stepSep

from .cm_models.passive_cm_models import P_RW_0step, P_RW_1stepDecay, P_RW_1stepMean, P_RW_1stepSep, P_RW_2stepDecay, P_RW_2stepMean, P_RW_2stepSep, P_RWSep_0step, P_RWSep_1stepDecay, P_RWSep_1stepMean, P_RWSep_1stepSep, P_RWSep_2stepDecay, P_RWSep_2stepMean, P_RWSep_2stepSep

from .cm_models.null_models import BiasedCEC, HeuCEC, RWCEC

## Batchfit class
class BatchFit(object):
    def __init__(self, model_list, longform, df_summary, project_dir, save_path, save_mood_path=None):
        self.models = {}
        for model_name in model_list:
            self.models[model_name] = eval(f'{model_name}(longform, df_summary, project_dir, save_path, save_mood_path)')
        self.longform = longform
        self.df_summary = df_summary
        self.project_dir = project_dir
    
    def fit(self, pid_num, block, jupyter=False):
        for model_name in self.models:
            if jupyter:
                clear_output(wait=True)
            print(f'Fitting {model_name}: PID - {pid_num}, Block - {block}')
            self.models[model_name].fit(pid_num, block)
    
    def fit_mood(self, pid_num, block, jupyter=False):
        for model_name in self.models:
            if jupyter:
                clear_output(wait=True)
            print(f'Fitting {model_name}: PID - {pid_num}, Block - {block}')
            self.models[model_name].fit_mood(pid_num, block)
    
    def join(self, batch):
        for model_name in batch.models:
            if model_name not in self.models:
                self.models[model_name] = batch.models[model_name]
            else:
                print(f'Model {model_name} already exists in this batch')
                
