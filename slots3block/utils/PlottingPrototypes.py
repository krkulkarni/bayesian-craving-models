import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DecisionModelPlotting(object):
    
    @classmethod
    def plot_choices(cls, act_rew_rate, adjust_size=(1000,1000), streamlit=False):
    
        figs = []    
        for block in ['positive', 'negative', 'mixed']:
            fig = go.Figure()
            for y, sub_choices in enumerate(act_rew_rate[block]['actions']):
                right_index = sub_choices.astype(bool)
                left_index = right_index==False
                right_choices = np.arange(len(sub_choices))[right_index]
                left_choices = np.arange(len(sub_choices))[left_index]

                fig.add_trace(go.Scatter(x=right_choices, y=[y]*len(right_choices),
                                        mode='markers', showlegend=False,
                                        marker=dict(color='cornflowerblue', symbol='triangle-right', size=12)))

                fig.add_trace(go.Scatter(x=left_choices, y=[y]*len(left_choices),
                                        mode='markers', showlegend=False,
                                        marker=dict(color='orange', symbol='triangle-left', size=11)))
                height, width = adjust_size
                fig.update_layout(
                    title=f'{block.capitalize()} Block Choices',
                    autosize=False,
                    width=width,
                    height=height,
                )
            figs.append(fig)

        if streamlit:
            return figs
        else:
            for fig in figs:
                fig.show()

    @classmethod
    def plot_choice_accuracy(cls, act_rew_rate, 
            adjust_size=(500,500), streamlit=False):

        fig = go.Figure()

        num_trials = len(act_rew_rate['positive']['actions'][0])
        num_participants = act_rew_rate['pids'].shape[0]
        colors = ['indianred', 'lightseagreen', 'cornflowerblue']
        below_chance = []

        accs = []

        for b, block in enumerate(['positive', 'negative', 'mixed']):
            acc = np.zeros(num_participants)
            
            for i, sub_choices in enumerate(act_rew_rate[block]['actions']):
                
                if act_rew_rate['reversal_timings'][i]=='A' or act_rew_rate['reversal_timings'][i]=='C':
                    timings = np.array([12, 12, 11])
                elif act_rew_rate['reversal_timings'][i]=='B' or act_rew_rate['reversal_timings'][i]=='D':
                    timings = np.array([13, 12, 10])
                reward_structure = np.array([np.array([0.8]*timings[0] + [0.2]*timings[1] + [0.8]*timings[2]),
                                        np.array([0.2]*timings[0] + [0.8]*timings[1] + [0.2]*timings[2])])

                optimal_choice = np.array(reward_structure[0,:] < reward_structure[1, :]).astype(int)
                if act_rew_rate[block]['block_num'][i]==2:
                    optimal_choice = 1 - optimal_choice

                acc[i] = np.sum((np.array(sub_choices)==optimal_choice).astype(int))/num_trials
                if acc[i] < 0.5:
                    # print(i, act_rew_rate['pids'][i], block)
                    below_chance.append(act_rew_rate['pids'][i])
            
            accs.append(acc)

            fig.add_trace(go.Box(
                    y=acc,
                    x=[b] * len(acc),
                    width=0.3,
                    name=block,
                    marker_color=colors[b],
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-0
                ))
        
        # Add line for chance
        fig.add_shape(type="line",
            x0=-0.25, y0=0.5, x1=2.25, y1=0.5,
            line=dict(color="RoyalBlue",width=3, dash='dot')
)

        height, width = adjust_size
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            boxmode='group',
            xaxis = dict(
                tickmode = 'array',
                tickvals = np.arange(len(['positive', 'negative', 'mixed'])),
                ticktext = ['positive', 'negative', 'mixed']
            )
        )

        fig.show()  
        return accs
        # return(np.array(set(below_chance)))
                
    
    @classmethod
    def plot_data(cls, actions, rewards, Qs):
        plt.figure(figsize=(20,3))
        x = np.arange(len(actions))

        if Qs.shape[0] > 1:
            norm_f = np.max(Qs)
            plt.plot(x, Qs[:, 0]/norm_f - .5 + 0, c='C0', lw=3, alpha=.3)
            plt.plot(x, Qs[:, 1]/norm_f - .5 + 1, c='C1', lw=3, alpha=.3)

        s = 50
        lw = 2

        # Left and no reward
        cond = (actions == 0) & (rewards == 0)
        plt.scatter(x[cond], actions[cond], s=s, c='None', ec='C0', lw=lw)

        # Left and reward
        cond = (actions == 0) & (rewards > 0)
        plt.scatter(x[cond], actions[cond], s=s, c='C0', ec='C0', lw=lw)

        # Right and no reward
        cond = (actions == 1) & (rewards == 0)
        plt.scatter(x[cond], actions[cond], s=s, c='None', ec='C1', lw=lw)

        # Right and reward
        cond = (actions == 1) & (rewards > 0)
        plt.scatter(x[cond], actions[cond], s=s, c='C1', ec='C1', lw=lw)

        plt.scatter(0, 20, c='k', s=s, lw=lw, label='Reward')
        plt.scatter(0, 20, c='w', ec='k', s=s, lw=lw, label='No reward')
        plt.plot([0,1], [20, 20], c='k', lw=3, alpha=.3, label='Qvalue (centered)')

        plt.yticks([0,1], ['left', 'right'])
        plt.ylim(-1, 2)

        plt.ylabel('action')
        plt.xlabel('trial')

        handles, labels = plt.gca().get_legend_handles_labels()
        order = (1,2,0)
        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]

        plt.legend(handles, labels, fontsize=12, loc=(1.01, .27))
        plt.tight_layout()

        
    @classmethod
    def plot_parameters(cls, fit_metrics, update_to_range=None, adjust_size=(500,800), chosen_params=None,
                        streamlit=False, true_fit_metrics=None, true_participant_num=None,
                        true_params=None):
        fig = go.Figure()
        block_colors = ['cornflowerblue', 'indianred', 'lightseagreen']

        for b, block in enumerate(fit_metrics.keys()):
            if block=='num_participants':
                continue
            if chosen_params is None:
                chosen_params = fit_metrics[block]['params'].keys()
            for param, values in fit_metrics[block]['params'].items():
                if param in chosen_params:
                    fig.add_trace(go.Box(
                        y=values,
                        x=[param]*len(values),
                        name=block,
                        marker_color=block_colors[b-1],
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-0
                    ))

        names = set()
        fig.for_each_trace(
            lambda trace:
                trace.update(showlegend=False)
                if (trace.name in names) else names.add(trace.name))
        
        height, width = adjust_size
        # labels = list(fit_metrics[block]['params'].keys())
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            boxmode='group',
            # xaxis = dict(
            #     tickmode = 'array',
            #     tickvals = np.arange(len(labels)),
            #     ticktext = labels
            # )
        )

        if update_to_range:
            fig.update_yaxes(range=update_to_range)

        if streamlit:
            return fig
        else:
            fig.show()

    @classmethod
    def model_comparison(cls, fit_metrics, labels, sums=False, update_to_range=None, 
            adjust_size=(700,1000), streamlit=False):
        
        bic_metrics = {key: {} for key in fit_metrics[0].keys() if key!='num_participants'}
        colors = ['cornflowerblue', 'indianred', 'lightseagreen']
        bar_width = 0.7/len(bic_metrics.keys())

        for block in bic_metrics.keys():
            bic_metrics[block]['bics'] = [elem[block]['bic'] for elem in fit_metrics]
            bic_metrics[block]['bic_sums'] = [np.sum(elem) for elem in bic_metrics[block]['bics']]


        fig = go.Figure()

        if not sums:
            for b, block in enumerate(bic_metrics.keys()):
                for values, label in zip(bic_metrics[block]['bics'], labels):
                    fig.add_trace(go.Box(
                        y=values,
                        x=[label]*len(values),
                        # width=bar_width,
                        name=block.capitalize(),
                        marker_color=colors[b],
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-0
                    ))

        
        if sums:
            for b, block in enumerate(bic_metrics.keys()):
                fig.add_trace(go.Bar(
                    y=bic_metrics[block]['bic_sums'],
                    x=labels,
                    width=bar_width,
                    name=block.capitalize(),
                    marker_color=colors[b]
                ))

        names = set()
        fig.for_each_trace(
            lambda trace:
                trace.update(showlegend=False)
                if (trace.name in names) else names.add(trace.name))
        
        height, width = adjust_size
        fig.update_layout(
            title='Model comparison (BIC)',
            xaxis_title='Model',
            yaxis_title='BIC Score',
            autosize=False,
            width=width,
            height=height,
            boxmode='group'
            # xaxis = dict(
            #     tickmode = 'array',
            #     tickvals = np.arange(len(labels)),
            #     ticktext = labels
            # )
        )

        if update_to_range:
            fig.update_yaxes(range=update_to_range)

        if streamlit:
            return fig
        else:
            fig.show()

    @classmethod
    def plot_values(cls, fit_metrics, labels, adjust_size=(700, 800), update_to_range=(-2, 4), streamlit=False):
        
        block_titles = [elem for elem in fit_metrics[0].keys() if elem!='num_participants']
        fig = make_subplots(rows=len(labels), cols=len(fit_metrics[0].keys())-1, row_titles=labels, column_titles=block_titles)
        for i, model_fm in enumerate(fit_metrics):
            if labels[i]=='random':
                continue
            for b, block in enumerate(model_fm.keys()):
                if block=='num_participants':
                    continue
                for values in np.array(model_fm[block]['values']):
                    # print(block)
                    # print(model_fm[block])
                    fig.add_trace(go.Scatter(
                        x=np.arange(values.shape[0]),
                        y=values[:,0],
                        opacity=0.08,
                        line=dict(color='indianred')
                    ), row=i+1, col=b)
                    fig.add_trace(go.Scatter(
                        x=np.arange(values.shape[0]),
                        y=values[:,1]+1,
                        opacity=0.08,
                        line=dict(color='lightseagreen')
                    ), row=i+1, col=b)
                    fig.update_yaxes(range=update_to_range, row=i+1, col=b+1)
        fig.update_layout(
            width=adjust_size[0],
            height=adjust_size[1],
            showlegend=False,
            title='Values'
        )
        
        if streamlit:
            return fig
        else:
            fig.show()
        
# class CravingModelPlotting(object):

#     @classmethod
#     def visualize_ratings(cls, act_rew_rate, streamlit=False):
    
#         fig = make_subplots(rows=1, cols=2, subplot_titles=['Money Ratings', 'Other Ratings'])

#         for y, sub_ratings in enumerate(act_rew_rate['money']['craving_ratings']):
#             x = np.arange(len(sub_ratings)) 

#             fig.add_trace(go.Scatter(x=x, y=[y]*len(sub_ratings),
#                                     mode='markers', showlegend=False,
#                                     marker=dict(color=-1*sub_ratings, cmin=-10, cmax=-1, size=sub_ratings*2.5,  colorscale='RdBu')),
#                         row=1, col=1)

#         for y, sub_ratings in enumerate(act_rew_rate['other']['craving_ratings']):
#             x = np.arange(len(sub_ratings)) 

#             fig.add_trace(go.Scatter(x=x, y=[y]*len(sub_ratings),
#                                     mode='markers', showlegend=False,
#                                     marker=dict(color=-1*sub_ratings, cmin=-10, cmax=-1, size=sub_ratings*2.5,  colorscale='RdBu')),
#                         row=1, col=2)

#             fig.update_layout(
#                 autosize=False,
#                 width=1200,
#                 height=900,
#             )

#         if streamlit:
#             return fig
#         else:
#             fig.show()
    
#     @classmethod
#     def plot_params(cls, fit_metrics, params=None, adjust_size=(500,800), 
#                 update_to_range=None, streamlit=False):
    
#         fig = go.Figure()

#         if params is None or params=='all':
#             params = list(fit_metrics['money']['params'].keys())

#         for i, param in enumerate(params):
#             values = fit_metrics['money']['params'][param]
#             fig.add_trace(go.Box(
#                 y=values,
#                 x=[i - 0.2]*len(values),
#                 name='Money',
#                 marker_color='indianred',
#                 boxpoints='all',
#                 jitter=0.3,
#                 pointpos=-0
#             ))

#         for i, param in enumerate(params):
#             values = fit_metrics['other']['params'][param]
#             fig.add_trace(go.Box(
#                 y=values,
#                 x=[i + 0.2]*len(values),
#                 name='Other',
#                 marker_color='lightseagreen',
#                 boxpoints='all',
#                 jitter=0.3,
#                 pointpos=-0
#             ))

#         names = set()
#         fig.for_each_trace(
#             lambda trace:
#                 trace.update(showlegend=False)
#                 if (trace.name in names) else names.add(trace.name))

#         if update_to_range:
#             fig.update_yaxes(range=update_to_range)

#         height, width = adjust_size
#         fig.update_layout(
#             autosize=False,
#             height=height, width=width,
#             xaxis = dict(
#                     tickmode = 'array',
#                     tickvals = np.arange(len(params)),
#                     ticktext = params
#                 )
#         )

#         if streamlit:
#             return fig
#         else:
#             fig.show()
        
        
#     @classmethod
#     def plot_bic(cls, fit_metrics, update_to_range=None, adjust_size=(500,600)):

#         fig = go.Figure()

#         fig.add_trace(go.Box(
#             y=fit_metrics['money']['bic'],
#             x=['Money']*len(fit_metrics['money']['bic']),
#             name='Money',
#             marker_color='lightseagreen',
#             boxpoints='all',
#             jitter=0.3,
#             pointpos=-0
#         ))

#         fig.add_trace(go.Box(
#             y=fit_metrics['other']['bic'],
#             x=['Other']*len(fit_metrics['money']['bic']),
#             name='Other',
#             marker_color='indianred',
#             boxpoints='all',
#             jitter=0.3,
#             pointpos=-0
#         ))  

#         if update_to_range:
#             fig.update_yaxes(range=update_to_range)
        
#         height, width = adjust_size
#         fig.update_layout(
#             autosize=False,
#             height=height, width=width,
#             xaxis = dict(
#                     tickmode = 'array',
#                     tickvals = np.arange(len(['Money', 'other'])),
#                     ticktext = ['Money', 'Other']
#                 )
#         )

#         fig.show()
        
    
#     @classmethod
#     def model_comparison(cls, fit_metrics, labels, sums=False, update_to_range=None, 
#         adjust_size=(700,1000), streamlit=False):

#         money_bics = [elem['money']['bic'] for elem in fit_metrics]
#         money_bics_sums = [np.sum(elem) for elem in money_bics]
#         other_bics = [elem['other']['bic'] for elem in fit_metrics]
#         other_bics_sums = [np.sum(elem) for elem in other_bics]

#         fig = go.Figure()

#         if not sums:
#             for i, values in enumerate(money_bics):
#                 fig.add_trace(go.Box(
#                     y=values,
#                     x=[i-0.2]*len(values),
#                     width=0.2,
#                     name='Money',
#                     marker_color='indianred',
#                     boxpoints='all',
#                     jitter=0.3,
#                     pointpos=-0
#                 ))

#             for i, values in enumerate(other_bics):
#                 fig.add_trace(go.Box(
#                     y=values,
#                     x=[i+0.2]*len(values),
#                     width=0.2,
#                     name='Other',
#                     marker_color='lightseagreen',
#                     boxpoints='all',
#                     jitter=0.3,
#                     pointpos=-0
#                 ))

#             names = set()
#             fig.for_each_trace(
#                 lambda trace:
#                     trace.update(showlegend=False)
#                     if (trace.name in names) else names.add(trace.name))
        
#         if sums:
#             fig.add_trace(go.Bar(
#                 y=money_bics_sums,
#                 x=np.arange(len(money_bics_sums))-0.2,
#                 width=0.3,
#                 name='Money',
#                 marker_color='indianred'
#             ))
            
#             fig.add_trace(go.Bar(
#                 y=other_bics_sums,
#                 x=np.arange(len(money_bics_sums))+0.2,
#                 width=0.3,
#                 name='Other',
#                 marker_color='lightseagreen'
#             ))
            

#         height, width = adjust_size
#         fig.update_layout(
#             title='Model comparison (BIC)',
#             xaxis_title='Model',
#             yaxis_title='BIC Score',
#             autosize=False,
#             width=width,
#             height=height,
#     #         boxmode='group',
#             xaxis = dict(
#                 tickmode = 'array',
#                 tickvals = np.arange(len(labels)),
#                 ticktext = labels
#             )
#         )

#         if update_to_range:
#             fig.update_yaxes(range=update_to_range)

#         if streamlit:
#             return fig
#         else:
#             fig.show()