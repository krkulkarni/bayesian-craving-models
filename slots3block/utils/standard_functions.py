import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import glob
import copy

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data(path_to_data, num_trials=60, num_craving_ratings=20, num_mood_ratings=12):
    
    # Load all csvs into li
    all_csvs = glob.glob(path_to_data + "*.csv")
    li = []
    for filename in all_csvs:
        df = pd.read_csv(filename)
        li.append(df)
        
    # Create dictionary of data
    act_rew_rate = {
        'pids': [],
        'reversal_timings': [],
        'first_block': [],
        'money': {
            'actions': np.zeros((len(li), num_trials)),
            'rewards': np.zeros((len(li), num_trials)),
            'craving_ratings': np.zeros((len(li), num_craving_ratings)),
            'mood_ratings': np.zeros((len(li), num_mood_ratings)),
        },
        'other': {
            'actions': np.zeros((len(li), num_trials)),
            'rewards': np.zeros((len(li), num_trials)),
            'craving_ratings': np.zeros((len(li), num_craving_ratings)),
            'mood_ratings': np.zeros((len(li), num_mood_ratings)),
        }
    }

    for i, subject in enumerate(li):
        # Obtain PIDs
        instructions_data = subject[subject['phase']=='instructions']
        block_order = np.array(eval(instructions_data['block_order'].to_numpy()[0]))
        practice_data = subject[subject['phase']=='practice']
        money_data = subject[subject['phase']=='money']
        other_data = subject[subject['phase']=='other']

        act_rew_rate['pids'].append(instructions_data['prolific_id'].to_numpy()[0])
        act_rew_rate['reversal_timings'].append(instructions_data['reversal_timings'][0])
        act_rew_rate['first_block'].append(block_order[0])
        
        # Obtain money choices
        choices = np.array(eval(money_data['choices'].to_numpy()[0]))
        if instructions_data['reversal_timings'][0]=='B' or instructions_data['reversal_timings'][0]=='D':
            choices = 1 - choices
        if block_order[0]=='other':
            choices = 1 - choices
        # choices = np.array([0 if elem=='L' else 1 for elem in letter_choices])
        act_rew_rate['money']['actions'][i,:] = choices

        # Obtain money rewards
        rewards = np.array(eval(money_data['outcomes'].to_numpy()[0]))
        act_rew_rate['money']['rewards'][i,:] = np.array(rewards)

        # Obtain money ratings
        act_rew_rate['money']['craving_ratings'][i,:] = np.array(eval(money_data['craving_ratings'].to_numpy()[0]))
        act_rew_rate['money']['mood_ratings'][i,:] = np.array(eval(money_data['mood_ratings'].to_numpy()[0]))

        # Obtain other choices
        try:
            choices = np.array(eval(other_data['choices'].to_numpy()[0]))
        except KeyError:
            print(subject)
            raise(ValueError)
        if instructions_data['reversal_timings'][0]=='B' or instructions_data['reversal_timings'][0]=='D':
            choices = 1 - choices
        if block_order[0]=='other':
            choices = 1 - choices
        # choices = np.array([0 if elem=='L' else 1 for elem in letter_choices])
        act_rew_rate['other']['actions'][i,:] = choices

        # Obtain other rewards
        rewards = np.array(eval(other_data['outcomes'].to_numpy()[0]))
        act_rew_rate['other']['rewards'][i,:] = np.array(rewards)

        # Obtain other ratings
        act_rew_rate['other']['craving_ratings'][i,:] = np.array(eval(other_data['craving_ratings'].to_numpy()[0]))
        act_rew_rate['other']['mood_ratings'][i,:] = np.array(eval(other_data['mood_ratings'].to_numpy()[0]))
        
    act_rew_rate['pids'] = np.array(act_rew_rate['pids'])
    act_rew_rate['reversal_timings'] = np.array(act_rew_rate['reversal_timings'])
    act_rew_rate['first_block'] = np.array(act_rew_rate['first_block'])
    act_rew_rate['money']['actions'] = act_rew_rate['money']['actions'].astype(int)
    act_rew_rate['other']['actions'] = act_rew_rate['other']['actions'].astype(int)
    act_rew_rate['money']['craving_ratings'] = act_rew_rate['money']['craving_ratings'].astype(int)
    act_rew_rate['other']['craving_ratings'] = act_rew_rate['other']['craving_ratings'].astype(int)
    act_rew_rate['money']['mood_ratings'] = act_rew_rate['money']['mood_ratings'].astype(int)
    act_rew_rate['other']['mood_ratings'] = act_rew_rate['other']['mood_ratings'].astype(int)

    # Define the rating trials
    # rating_trials = {
    #     'money': np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 
    #         0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1]),
    #     'other': np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 
    #         0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    # }
    
    # Define the fast and slow spin trials
    # delays = {
    #     'money': np.array([1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 
    #         1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2]) - 1,
    #     'other': np.array([1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 
    #         2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1]) - 1
    # }
    
    return act_rew_rate
    # return act_rew_rate, rating_trials, delays


def subset(orig_act_rew_rate, subset_ind):
    
    act_rew_rate = copy.deepcopy(orig_act_rew_rate)
    
    act_rew_rate['pids'] = act_rew_rate['pids'][subset_ind]
    
    act_rew_rate['money']['actions'] = act_rew_rate['money']['actions'][subset_ind,:]
    act_rew_rate['money']['rewards'] = act_rew_rate['money']['rewards'][subset_ind,:]
    act_rew_rate['money']['ratings'] = act_rew_rate['money']['ratings'][subset_ind,:]
    
    act_rew_rate['other']['actions'] = act_rew_rate['other']['actions'][subset_ind,:]
    act_rew_rate['other']['rewards'] = act_rew_rate['other']['rewards'][subset_ind,:]
    act_rew_rate['other']['ratings'] = act_rew_rate['other']['ratings'][subset_ind,:]
    
    return act_rew_rate


def load_questionnaire_data(path_to_qdata, pids, fields=None):
    redcap_csv_path = glob.glob(path_to_qdata)[0]
    redcap_df = pd.read_csv(redcap_csv_path)
    
    if fields is None:
        fields = ['prolific_pid', 'gi_1', 'gi_2', 'mshq_1', 'mshq_2',
       'mshq_3', 'mshq_4', 'mshq_5', 'mshq_6', 'mshq_7', 'mshq_8', 'mshq_9',
       'mshq_10', 'mshq_11', 'mshq_12', 'mshq_13', 'mshq_14', 'mshq_15',
       'mshq_16', 'mshq_17', 'mshq_18', 'mshq_19', 'mshq_20', 'mshq_21', 
       'mps_1', 'mps_2', 'mps_3', 'mps_4', 'mps_5', 'mps_6', 'mps_7', 
       'mps_8', 'mps_9', 'mps_10', 'mps_11', 'mps_12', 'mps_13', 'mps_14', 
       'mps_15', 'mps_16', 'mps_17', 'mps_18', 'mps_19']
        
    redcap_df = redcap_df.filter(items=fields)
    
    valid_redcap_df = redcap_df[redcap_df['prolific_pid'].isin(pids)]
    valid_redcap_df = valid_redcap_df.set_index('prolific_pid').loc[pids]
    
    return valid_redcap_df


##########################################
# PLOTTING FUNCTIONS #
##########################################

# def visualize_ratings(act_rew_rate):
    
#     fig = make_subplots(rows=1, cols=2, subplot_titles=['Money Ratings', 'Other Ratings'])

#     for y, sub_ratings in enumerate(act_rew_rate['money']['ratings']):
#         x = np.arange(len(sub_ratings)) 

#         fig.add_trace(go.Scatter(x=x, y=[y]*len(sub_ratings),
#                                 mode='markers', showlegend=False,
#                                 marker=dict(color=-1*sub_ratings, cmin=-10, cmax=-1, size=sub_ratings*2.5,  colorscale='RdBu')),
#                      row=1, col=1)

#     for y, sub_ratings in enumerate(act_rew_rate['other']['ratings']):
#         x = np.arange(len(sub_ratings)) 

#         fig.add_trace(go.Scatter(x=x, y=[y]*len(sub_ratings),
#                                 mode='markers', showlegend=False,
#                                 marker=dict(color=-1*sub_ratings, cmin=-10, cmax=-1, size=sub_ratings*2.5,  colorscale='RdBu')),
#                      row=1, col=2)

#         fig.update_layout(
#             autosize=False,
#             width=1200,
#             height=900,
#         )

#     fig.show()