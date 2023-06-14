
## Latest version of the models for the slotscraving task
## Author: Kaustubh Kulkarni
## Original Date: June 2022
## Last Modified: May 1, 2023

import numpy as np
from scipy import stats
import pymc as pm
import aesara.tensor as at
import aesara
import arviz as az
from .prototypes_cm import MixedPrototype_Beta, MixedPrototype, MixedPrototype_noCEC, MixedPrototype_Beta_noCEC
import os

"""
    Note: all models have the following features:
        1. Asymmetric learning from positive and negative prediction errors
        2. 2-step look into history of prediction errors and values (decaying effect)

    Models for the 3x3 factorial design
    3x3 factorial design: 3 types of decision models x 3 types of craving models

    Decision models:
        1. Rescorla Wagner - No craving modulation (NoBias)
        2. Rescorla Wagner - Craving modulation of learning rate (LRBias)
        3. Rescorla Wagner - Craving modulation of temperature (TempBias)

    Craving models:
        1. Cue-elicited craving (CEC)
        2. Expectation-elicited craving (EEC)
        3. Cue-elicted craving + Expectation-elicited craving (JEC)

    Note: The models are defined in the following order:
        1. NoBias_CEC
        2. NoBias_EEC
        3. NoBias_JEC
        4. LRBias_CEC
        5. LRBias_EEC
        6. LRBias_JEC
        7. TempBias_CEC
        8. TempBias_EEC
        9. TempBias_JEC
        10. Heu_CEC (Null model; Decisions are made with a win-stay-lose-shift heuristic, craving is cue-elicited)
"""

## 1. NoBias_CEC
class NoBias_CEC(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'NoBias_CEC'
        self.decision = 'NoBias'
        self.craving = 'CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero

## 2. NoBias_EEC
class NoBias_EEC(MixedPrototype_noCEC):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'NoBias_EEC'
        self.decision = 'NoBias'
        self.craving = 'EEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        return beta, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

## 3. NoBias_JEC
class NoBias_JEC(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'NoBias_JEC'
        self.decision = 'NoBias'
        self.craving = 'JEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + pos_al * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

## 4. LRBias_CEC
class LRBias_CEC(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'LRBias_CEC'
        self.decision = 'LRBias'
        self.craving = 'CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        mod = pm.Normal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, mod

## 5. LRBias_EEC
class LRBias_EEC(MixedPrototype_noCEC):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'LRBias_EEC'
        self.decision = 'LRBias'
        self.craving = 'EEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.Normal('mod', mu=0, sigma=0.5)
        return beta, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, mod

## 6. LRBias_JEC
class LRBias_JEC(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'LRBias_JEC'
        self.decision = 'LRBias'
        self.craving = 'JEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, mod = args

        ## Calculate the bias associated with the current craving
        bias = mod*pm.math.invlogit(Qs[2])

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + (neg_al + bias) * (r - Qs[a])),
            at.set_subtensor(Qs[a], Qs[a] + (pos_al + bias) * (r - Qs[a]))
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.Normal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, mod

## 7. TempBias_CEC
class TempBias_CEC(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'TempBias_CEC'
        self.decision = 'TempBias'
        self.craving = 'CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        mod = pm.Normal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero

## 8. TempBias_EEC
class TempBias_EEC(MixedPrototype_Beta_noCEC):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'TempBias_EEC'
        self.decision = 'TempBias'
        self.craving = 'EEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.Normal('mod', mu=0, sigma=0.5)
        return beta, mod, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

## 9. TempBias_JEC
class TempBias_JEC(MixedPrototype_Beta):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'TempBias_JEC'
        self.decision = 'TempBias'
        self.craving = 'JEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = aesara.ifelse.ifelse(
            at.lt(r-Qs[a], 0),
            at.set_subtensor(Qs[a], Qs[a] + neg_al*pe),
            at.set_subtensor(Qs[a], Qs[a] + pos_al*pe)
        )

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + (w1**3)*Qs[4] + w2*pe + (w2**2)*Qs[5] + (w2**3)*Qs[6]
        )

        # Set PE[t-2] at index 6 with PE[t-1] at index 5
        Qs = at.set_subtensor(Qs[6], Qs[5])
        # Set PE[t-1] at index 5 with trial PE
        Qs = at.set_subtensor(Qs[5], pe)
        # Set Qs[t-2] at index 4 with Qs[t-1] at index 3
        Qs = at.set_subtensor(Qs[4], Qs[3])
        # Set Qs[t-1] at index 3 with trial chosen Q
        Qs = at.set_subtensor(Qs[3], Qs[a])
        
        return Qs

    def _define_priors(self):
        beta = pm.Exponential('beta', 0.75)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        mod = pm.Normal('mod', mu=0, sigma=0.5)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, mod, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

## 10. Heu_CEC
class Heu_CEC:
    def __init__(self, longform, summary, project_dir, save_path, save_mood_path):
        self.name = 'Heu_CEC'        
        self.longform = longform
        self.summary = summary
        self.pid_list = longform['PID'].unique()
        self.traces = {
            'money': {},
            'other': {}
        }
        self.mood_traces = {
            'money': {},
            'other': {}
        }
        self.project_dir = project_dir

        self.decision = 'Heu'
        self.craving = 'CEC'

        num_craving_trials = 20
        num_mood_trials = 12
        num_blocks = 2
        
        self.craving_inds = None
        self.mean_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.std_craving = np.zeros((num_blocks, len(self.pid_list)))
        self.norm_cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))
        self.cravings = np.zeros((num_blocks, len(self.pid_list), num_craving_trials))

        self.mood_inds = None
        self.mean_mood = np.zeros((num_blocks, len(self.pid_list)))
        self.std_mood = np.zeros((num_blocks, len(self.pid_list)))
        self.norm_moods = np.zeros((num_blocks, len(self.pid_list), num_mood_trials))
        self.moods = np.zeros((num_blocks, len(self.pid_list), num_mood_trials))

        self._calc_norm_cravings_moods()

        self.save_path = save_path
        self.save_mood_path = save_mood_path
    
    def _calc_norm_cravings_moods(self):
        for pid_num in range(len(self.pid_list)):
            for b, block in enumerate(['money', 'other']):
                pid = self.pid_list[pid_num]
                cravings = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Craving Rating'].values
                moods = self.longform[(self.longform['PID']==pid)&(self.longform['Type']==block)]['Mood Rating'].values
                craving_inds = np.squeeze(np.argwhere(cravings>-1))
                mood_inds = np.squeeze(np.argwhere(moods>-1))

                mask = np.ones(len(craving_inds), dtype=bool)
                mask[12] = False
                craving_inds = craving_inds[mask]
                cravings = cravings[craving_inds]
                moods = moods[mood_inds]
                
                self.craving_inds = craving_inds
                self.mean_craving[b, pid_num] = np.mean(cravings)
                self.std_craving[b, pid_num] = np.std(cravings)
                self.norm_cravings[b, pid_num, :] = stats.zscore(cravings)
                self.cravings[b, pid_num, :] = cravings
                
                self.mood_inds = mood_inds
                self.mean_mood[b, pid_num] = np.mean(moods)
                self.std_mood[b, pid_num] = np.std(moods)
                self.norm_moods[b, pid_num, :] = stats.zscore(moods)
                self.moods[b, pid_num, :] = moods

    def add_eps(self, st, a, eps_t, eps):
        return aesara.ifelse.ifelse(
                at.eq(st, 1),
                aesara.ifelse.ifelse(
                    at.eq(a, 1),
                    eps,
                    1-eps
                ),
                aesara.ifelse.ifelse(
                    at.eq(a, 1),
                    1-eps,
                    eps
                )
            )
    
    def right_action_probs(self, actions, rewards, strat, *args):
        
        sample_eps, weight_zero, weight_one = args

        t_strat = at.as_tensor_variable(strat, dtype='int32')
        t_actions = at.as_tensor_variable(actions, dtype='int32')
        t_rewards = at.as_tensor_variable(rewards, dtype='int32')

        # Compute the Qs values
        t_eps = at.as_tensor_variable(np.asarray(1, 'float64'))
        t_eps, updates = aesara.scan(
            fn=self.add_eps,
            sequences=[t_strat, t_actions],
            outputs_info=t_eps,
            non_sequences=[sample_eps])
        
        pred_cravings = weight_zero + weight_one*t_rewards

        return t_eps[1:],  pm.invlogit(pred_cravings)
    
    def _load_act_rew_craving_mood(self, pid_num, block, norm=True):
        pid = self.pid_list[pid_num]
        b = 0 if block=='money' else 1
        act = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Action'].values
        rew = self.longform[(self.longform['PID']==pid) & (self.longform['Type']==block)]['Reward'].values
        if norm:
            crav = self.norm_cravings[b, pid_num, :]
            mood = self.norm_moods[b, pid_num, :]
        else:
            crav = self.cravings[b, pid_num, :]
            mood = self.moods[b, pid_num, :]
        return act, rew, crav, mood
    
    def _define_priors(self):
        untr_eps = pm.Normal('untr_eps', mu=0, sigma=1)
        eps = pm.Deterministic('eps', pm.math.invlogit(untr_eps))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        return eps, weight_zero, weight_one
        

    def fit(self, pid_num, block):
        pid = self.pid_list[pid_num]
        if self.save_path is not None:
            if not os.path.exists(f'{self.save_path}/{self.name}/'):
                os.makedirs(f'{self.save_path}/{self.name}/')
            filestr = f'{self.save_path}/{self.name}/{block}_{pid}.nc'
            if os.path.exists(filestr):
                print(f'PID: {pid}, Block {block} exists, loading from file...')
                self.traces[block][pid] = az.from_netcdf(filestr)
                return
        act, rew, cravings, moods = self._load_act_rew_craving_mood(pid_num, block, norm=False)
        strat = np.zeros(len(act))
        for i, a in enumerate(act):
            if i < 2:
                continue
            should_switch = np.all(np.array([rew[i-2]==rew[i-1], rew[i-1]==0]), axis=0)
            do_switch = act[i-1]!=act[i]
            strat[i] = should_switch==do_switch
        strat = strat.astype(int)
        with pm.Model() as model:
            priors = self._define_priors()
            action_probs, craving_pred = self.right_action_probs(act, rew, strat, *priors)

            like = pm.Bernoulli('like', p=action_probs, observed=act[1:])
            probs_craving = pm.Deterministic('probs_craving', craving_pred[self.craving_inds-1])
            craving_like = pm.Binomial('craving_like', n=50, p=probs_craving, observed=cravings)
            
            self.traces[block][pid] = pm.sample(step=pm.Metropolis())
            # self.traces[block][pid].extend(pm.sample_prior_predictive())
            pm.sample_posterior_predictive(self.traces[block][pid], extend_inferencedata=True)
            if self.save_path is not None:
                self.traces[block][pid].to_netcdf(filestr)
    
    def fit_mood(self, pid_num, block):
        if self.save_mood_path is None:
            return 'No mood save path specified'
        pid = self.pid_list[pid_num]
        if self.save_mood_path is not None:
            if not os.path.exists(f'{self.save_mood_path}/{self.name}/'):
                os.makedirs(f'{self.save_mood_path}/{self.name}/')
            filestr = f'{self.save_mood_path}/{self.name}/{block}_{pid}.nc'
            if os.path.exists(filestr):
                print(f'PID: {pid}, Block {block} exists, loading from file...')
                self.mood_traces[block][pid] = az.from_netcdf(filestr)
                return
        act, rew, cravings, moods = self._load_act_rew_craving_mood(pid_num, block, norm=False)
        strat = np.zeros(len(act))
        for i, a in enumerate(act):
            if i < 2:
                continue
            should_switch = np.all(np.array([rew[i-2]==rew[i-1], rew[i-1]==0]), axis=0)
            do_switch = act[i-1]!=act[i]
            strat[i] = should_switch==do_switch
        strat = strat.astype(int)
        with pm.Model() as model:
            priors = self._define_priors()
            action_probs, mood_pred = self.right_action_probs(act, rew, strat, *priors)

            like = pm.Bernoulli('like', p=action_probs, observed=act[1:])
            probs_mood = pm.Deterministic('probs_mood', mood_pred[self.mood_inds-1])
            mood_like = pm.Binomial('mood_like', n=50, p=probs_mood, observed=moods)
            
            self.mood_traces[block][pid] = pm.sample(step=pm.Metropolis())
            # self.mood_traces[block][pid].extend(pm.sample_prior_predictive())
            pm.sample_posterior_predictive(self.mood_traces[block][pid], extend_inferencedata=True)
            if self.save_mood_path is not None:
                self.mood_traces[block][pid].to_netcdf(filestr)
