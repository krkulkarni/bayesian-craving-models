import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# from scipy.special import logsumexp
# from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import pandas as pd


def plot_data(model, block_type, pid):
    actions = model.longform[
        (model.longform["PID"] == pid) & (model.longform["Type"] == block_type)
    ]["Action"].to_numpy()
    rewards = model.longform[
        (model.longform["PID"] == pid) & (model.longform["Type"] == block_type)
    ]["Reward"].to_numpy()
    Qs = model.longform[
        (model.longform["PID"] == pid) & (model.longform["Type"] == block_type)
    ][["Q_left", "Q_right"]].to_numpy()

    plt.figure(figsize=(20, 3))
    x = np.arange(len(actions))

    if Qs.shape[0] > 1:
        norm_f = np.max(Qs)
        plt.plot(x, Qs[:, 0] / norm_f - 0.5 + 0, c="C0", lw=3, alpha=0.3)
        plt.plot(x, Qs[:, 1] / norm_f - 0.5 + 1, c="C1", lw=3, alpha=0.3)

    s = 50
    lw = 2

    # Left and no reward
    cond = (actions == 0) & (rewards == 0)
    plt.scatter(x[cond], actions[cond], s=s, c="None", ec="C0", lw=lw)

    # Left and reward
    cond = (actions == 0) & (rewards > 0)
    plt.scatter(x[cond], actions[cond], s=s, c="C0", ec="C0", lw=lw)

    # Right and no reward
    cond = (actions == 1) & (rewards == 0)
    plt.scatter(x[cond], actions[cond], s=s, c="None", ec="C1", lw=lw)

    # Right and reward
    cond = (actions == 1) & (rewards > 0)
    plt.scatter(x[cond], actions[cond], s=s, c="C1", ec="C1", lw=lw)

    plt.scatter(0, 20, c="k", s=s, lw=lw, label="Reward")
    plt.scatter(0, 20, c="w", ec="k", s=s, lw=lw, label="No reward")
    plt.plot([0, 1], [20, 20], c="k", lw=3, alpha=0.3, label="Qvalue (centered)")

    plt.yticks([0, 1], ["left", "right"])
    plt.ylim(-1, 2)

    plt.ylabel("action")
    plt.xlabel("trial")

    handles, labels = plt.gca().get_legend_handles_labels()
    order = (1, 2, 0)
    handles = [handles[idx] for idx in order]
    labels = [labels[idx] for idx in order]

    plt.legend(handles, labels, fontsize=12, loc=(1.01, 0.27))
    plt.tight_layout()


def plot_model_comparison(
    models, metric="bic", sum=False, width=800, height=500, y_range=None
):
    fig = go.Figure()
    colors = ["indianred", "lightseagreen", "cornflowerblue", "goldenrod", "darkorange"]
    n_participants = len(models[0].ics)

    if sum:
        x = ["money", "other"]
        for i, model in enumerate(models):
            ic = [
                np.sum(np.hstack([model.ics[pid]['money'][metric].mean() for pid in model.ics.keys()])),
                np.sum(np.hstack([model.ics[pid]['other'][metric].mean() for pid in model.ics.keys()]))
            ]
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=ic,
                    name=f"{model.model_name}",
                    marker_color=colors[i],
                )
            )
        fig.update_layout(barmode="group")

    else:
        x = ["Money"] * n_participants + ["Other"] * n_participants
        for i, model in enumerate(models):
            ic = np.array([
                np.hstack([model.ics[pid]['money'][metric].mean() for pid in model.ics.keys()]),
                np.hstack([model.ics[pid]['other'][metric].mean() for pid in model.ics.keys()])
            ])
            fig.add_trace(
                go.Box(
                    y=np.hstack([ic[0,:], ic[1, :]]),
                    x=x,
                    boxpoints="all",
                    pointpos=0,
                    marker_color=colors[i],
                    name=model.model_name,
                )
            )

        fig.update_layout(boxmode="group")

    fig.update_layout(
        height=height, width=width,
    )
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    fig.show()

def plot_params(model, width=500, height=300):
    param_estimates = {
        'param': [],
        'value': [],
        'std': [],
        'block': []
    }
    for pid in model.traces:
        for var in model.traces[pid]['money'].posterior.data_vars:
            param_estimates['param'].append(var)
            param_estimates['value'].append(float(model.traces[pid]['money'].posterior[var].mean()))
            param_estimates['std'].append(float(model.traces[pid]['money'].posterior[var].std()))
            param_estimates['block'].append('money')
            param_estimates['param'].append(var)
            param_estimates['value'].append(float(model.traces[pid]['other'].posterior[var].mean()))
            param_estimates['std'].append(float(model.traces[pid]['other'].posterior[var].std()))
            param_estimates['block'].append('other')

    param_estimates['param'] = np.array(param_estimates['param'])
    param_estimates['value'] = np.array(param_estimates['value'])
    param_estimates['std'] = np.array(param_estimates['std'])
    param_estimates['block'] = np.array(param_estimates['block'])

    fig = px.box(param_estimates, x='param', y='value', color='block', width=width, height=height, points='all', notched=True)
    # fig, ax = plt.subplots(1,1,figsize=(width, height))
    # sns.boxplot(data=pd.DataFrame.from_dict(param_estimates), x='param', y='value', hue='block', ax=ax)
    return fig
