## Latest version of the models for the slotscraving task
## Author: Kaustubh Kulkarni
## Original Date: June 2022
## Last Modified: Feb 5, 2023

import pymc as pm
import aesara.tensor as at
import aesara
from .prototypes_cm import MixedPrototype

## Inheritance models (EVRPE only)
### RW Models - Passive
class P_RW_0step(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_0step'
        self.type = 'passive'
        self.retro = 0
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'

    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * Qs[a] + w2 * pe)

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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_1stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_1stepMean'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - ONE STEP MEAN (Retro1) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3])/2 + w2 * (pe+Qs[5])/2)

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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_1stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_1stepDecay'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + w2*pe + (w2**2)*Qs[5]
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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_1stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_1stepSep'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, w3, w4 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*pe + w4*Qs[5]
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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, weight_three, weight_four

class P_RW_2stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_2stepMean'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3]+Qs[4])/3 + w2 * (pe+Qs[5]+Qs[6])/3)

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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_2stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_2stepDecay'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two

class P_RW_2stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RW_2stepSep'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RW'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        alpha, w0, w1, w2, w3, w4, w5, w6 = args

        ## Update the Q values - Single update (RW)
        # Calculate trial RPE
        # Qs at indices 0 (left) and 1 (right)
        pe = r - Qs[a]
        Qs = at.set_subtensor(Qs[a], Qs[a] + alpha * (pe))

        ## Calculate predicted craving - TWO STEP (Retro2) - at index 2
        Qs = at.set_subtensor(
            Qs[2], 
            w0 + w1*Qs[a] + w2*Qs[3] + w3*Qs[4] + w4*pe + w5*Qs[5] + w6*Qs[6]
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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha = pm.Normal('untr_alpha', mu=0, sigma=1)
        alpha = pm.Deterministic('alpha', pm.math.invlogit(untr_alpha))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        weight_five = pm.Normal('weight_five', mu=0, sigma=1)
        weight_six = pm.Normal('weight_six', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha, weight_zero, weight_one, weight_two, weight_three, weight_four, weight_five, weight_six

### RWSep Models - Passive
class P_RWSep_0step(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_0step'
        self.type = 'passive'
        self.retro = 0
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'

    
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
        Qs = at.set_subtensor(Qs[2], w0 + w1 * Qs[a] + w2 * pe)

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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_1stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_1stepMean'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
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

        ## Calculate predicted craving - ONE STEP MEAN (Retro1) - at index 2
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3])/2 + w2 * (pe+Qs[5])/2)

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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_1stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_1stepDecay'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
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
            w0 + w1*Qs[a] + (w1**2)*Qs[3] + w2*pe + (w2**2)*Qs[5]
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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_1stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_1stepSep'
        self.type = 'passive'
        self.retro = 1
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, w3, w4 = args

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
            w0 + w1*Qs[a] + w2*Qs[3] + w3*pe + w4*Qs[5]
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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, weight_three, weight_four

class P_RWSep_2stepMean(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_2stepMean'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
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
        Qs = at.set_subtensor(Qs[2], w0 + w1 * (Qs[a]+Qs[3]+Qs[4])/3 + w2 * (pe+Qs[5]+Qs[6])/3)

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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_2stepDecay(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_2stepDecay'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two

class P_RWSep_2stepSep(MixedPrototype):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'P_RWSep_2stepSep'
        self.type = 'passive'
        self.retro = 2
        self.decision = 'RWSep'
        self.craving = 'EVRPE_CEC'
    
    def update_Q(self, a, r, Qs, *args):
        # Note that beta is not present in the args list
        pos_al, neg_al, w0, w1, w2, w3, w4, w5, w6 = args

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
            w0 + w1*Qs[a] + w2*Qs[3] + w3*Qs[4] + w4*pe + w5*Qs[5] + w6*Qs[6]
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
        beta = pm.HalfNormal('beta', 10)
        untr_alpha_pos = pm.Normal('untr_alpha_pos', mu=0, sigma=1)
        untr_alpha_neg = pm.Normal('untr_alpha_neg', mu=0, sigma=1)
        alpha_pos = pm.Deterministic('alpha_pos', pm.math.invlogit(untr_alpha_pos))
        alpha_neg = pm.Deterministic('alpha_neg', pm.math.invlogit(untr_alpha_neg))
        weight_zero = pm.Normal('weight_zero', mu=0, sigma=1)
        weight_one = pm.Normal('weight_one', mu=0, sigma=1)
        weight_two = pm.Normal('weight_two', mu=0, sigma=1)
        weight_three = pm.Normal('weight_three', mu=0, sigma=1)
        weight_four = pm.Normal('weight_four', mu=0, sigma=1)
        weight_five = pm.Normal('weight_five', mu=0, sigma=1)
        weight_six = pm.Normal('weight_six', mu=0, sigma=1)
        cec_weight = pm.Normal('cec_weight', mu=0, sigma=1)
        return beta, cec_weight, alpha_pos, alpha_neg, weight_zero, weight_one, weight_two, weight_three, weight_four, weight_five, weight_six