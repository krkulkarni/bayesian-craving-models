from matplotlib import colors
from matplotlib.pyplot import show
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import statsmodels.api as sm
from scipy.stats import linregress


COLORS = [
    "rgb(4, 135, 137)",
    "rgb(80, 61, 46)",
    "rgb(212, 77, 39)",
    "rgb(226, 167, 46)",
    "rgb(102, 95, 30)",
    "rgb(247, 96, 79)",
    "rgb(111, 185, 191)",
    "rgb(4, 135, 137)",
    "rgb(80, 61, 46)",
    "rgb(212, 77, 39)",
    "rgb(226, 167, 46)",
]

STANDARD_COLORS = [
    "cornflowerblue",
    "lightcoral",
    "indianred",
    "lightseagreen",
    "darkorchid",
    "darkturquoise",
    "darkseagreen",
    "darkslateblue",
    "darkmagenta",
    "darkkhaki",
    "darkcyan",
]


def _get_pid_params(batchfit, chosen_model, pid):
    money_ws = {}
    other_ws = {}
    for var in ["w0", "w1", "w2"]:
        money_ws[var] = []
        other_ws[var] = []
        try:
            for money_chain, other_chain in zip(
                batchfit.craving_models[chosen_model].traces[pid]["money"].posterior[f"{var}_mean"],
                batchfit.craving_models[chosen_model].traces[pid]["other"].posterior[f"{var}_mean"]
            ):
                money_ws[var].append(money_chain.values)
                other_ws[var].append(other_chain.values)
        except AttributeError:
            print(f"{pid} has no traces for {var}")
            return None
        except KeyError:
            print(f"{pid} has no traces for {var}")
            return None

        money_ws[var] = np.hstack(money_ws[var])
        other_ws[var] = np.hstack(other_ws[var])

    pid_df = pd.concat(
        [
            pd.DataFrame.from_dict(
                {
                    "pid": [pid] * len(money_ws["w0"]),
                    "w0": money_ws["w0"],
                    "w1": money_ws["w1"],
                    "w2": money_ws["w2"],
                    "block": ["money"] * len(money_ws["w0"]),
                }
            ),
            pd.DataFrame.from_dict(
                {
                    "pid": [pid] * len(other_ws["w0"]),
                    "w0": other_ws["w0"],
                    "w1": other_ws["w1"],
                    "w2": other_ws["w2"],
                    "block": ["other"] * len(other_ws["w0"]),
                }
            ),
        ]
    )
    return pid_df


def _get_corrs(batchfit, model_name):
    corrs = []
    for pid in batchfit.pid_subset:
        pid_corrs = []
        for block in ["money", "other"]:
            true = batchfit.craving_models[model_name].norm_craving_ratings[pid][block]
            preds = batchfit.craving_models[model_name].predictions[pid][block]
            if preds.all() == None:
                pid_corrs.append(0)
            else:
                pid_corrs.append(np.corrcoef(true, preds.mean(axis=0))[0, 1])
        corrs.append(np.array(pid_corrs))
    return np.array(corrs)


def _get_ics(batchfit, model_name, metric):
    money_ics = (
        batchfit.craving_models[model_name]
        .ic[batchfit.craving_models[model_name].ic["PID"].isin(batchfit.pid_subset)][
            f"Money {metric}"
        ]
        .values
    )

    other_ics = (
        batchfit.craving_models[model_name]
        .ic[batchfit.craving_models[model_name].ic["PID"].isin(batchfit.pid_subset)][
            f"Other {metric}"
        ]
        .values
    )
    return np.vstack([money_ics, other_ics])


def plot_sample_predictions(batchfit, pid, width=800, height=500):

    blocks = ["money", "other"]

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Money", "Other"])

    true_ratings = np.vstack([
        batchfit.craving_models[batchfit.model_subset[0]].norm_craving_ratings[pid]['money'],
        batchfit.craving_models[batchfit.model_subset[0]].norm_craving_ratings[pid]['other']
    ])

    for i, model_name in enumerate(batchfit.model_subset):
        model = batchfit.craving_models[model_name]
        for b, block in enumerate(blocks):
            # Add true labels
            fig.add_trace(
                go.Scatter(
                    x=np.arange(true_ratings.shape[1]),
                    y=true_ratings[b, :],
                    name=f"True ratings",
                    showlegend=True,
                    line=dict(color='black', width=4, dash='dot')
                ),
                row=b+1,
                col=1,
            )
            # Add mean predictions
            model_prediction = model.predictions[pid][block].mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(model_prediction)),
                    y=model_prediction,
                    line_color=STANDARD_COLORS[i],
                    name=model.craving_model_name,
                    showlegend=True,
                ),
                row=b+1,
                col=1,
            )

    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))


    fig.update_layout(height=height, width=width)
    fig.show()


def plot_model_comparison(
    batchfit, metric="DIC", sum=False, width=800, height=500, x_range=None,
):

    model_dict = batchfit.craving_models
    pid_subset = batchfit.pid_subset
    model_subset = batchfit.model_subset

    fig = go.Figure()

    if sum:
        x = ["Money", "Other"]
        for i, (model_name, model) in enumerate(model_dict.items()):
            if model_name not in model_subset:
                continue
            if pid_subset is not None:
                filtered_df = model.ic.loc[model.ic["PID"].isin(pid_subset)]
            else:
                filtered_df = model.ic
            ic = [
                filtered_df[f"Money {metric}"].sum(),
                filtered_df[f"Other {metric}"].sum(),
            ]
            fig.add_trace(
                go.Bar(
                    y=x,
                    x=ic,
                    name=f"{model.craving_model_name}",
                    marker_color=STANDARD_COLORS[i],
                    orientation="h",
                )
            )
        fig.update_layout(barmode="group")

    else:
        for i, (model_name, model) in enumerate(model_dict.items()):
            if model_name not in model_subset:
                continue
            if pid_subset is not None:
                filtered_df = model.ic.loc[model.ic["PID"].isin(pid_subset)]
            else:
                filtered_df = model.ic

            x = ["Money"] * filtered_df.shape[0] + ["Other"] * filtered_df.shape[0]
            ic = np.hstack(
                [
                    filtered_df[f"Money {metric}"].values,
                    filtered_df[f"Other {metric}"].values,
                ]
            )
            fig.add_trace(
                go.Box(
                    x=ic,
                    y=x,
                    boxpoints="all",
                    pointpos=0,
                    marker_color=STANDARD_COLORS[i],
                    name=model.craving_model_name,
                    orientation="h",
                )
            )

        fig.update_layout(boxmode="group")

    fig.update_layout(
        height=height, width=width,
    )
    fig.update_xaxes(title_text="IC")
    fig.update_yaxes(title_text="Condition")
    if x_range is not None:
        fig.update_xaxes(range=x_range)

    return fig


def plot_corr_ic(batchfit, model_name, metric="DIC", width=1000, height=500):
    corrs = _get_corrs(batchfit, model_name)
    ics = _get_ics(batchfit, model_name, metric)
    annotations = []
    corr_ic_df = pd.DataFrame.from_dict({
        "PID": batchfit.pid_subset,
        "Money Corr": corrs[:, 0],
        "Other Corr": corrs[:, 1],
        "Money IC": ics[0, :],
        "Other IC": ics[1, :],
    }).dropna()
    x_names = [f"{elem} - {np.argwhere(np.array(batchfit.pid_subset)==elem)[0][0]}" for elem in corr_ic_df['PID'].values]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=["IC", "Correlations"],
    )

    ## IC plot
    # Markers
    fig.add_trace(
        go.Scatter(
            x=corr_ic_df['Money IC'],
            y=corr_ic_df['Other IC'],
            mode="markers",
            name=model_name,
            hovertemplate=x_names,
            marker_color=STANDARD_COLORS[2],
        ),
        row=1,
        col=1,
    )
    # Trendline
    slope, intercept, r_value, p_value, std_err = linregress(corr_ic_df['Money IC'], corr_ic_df['Other IC'])
    fig.add_trace(
        go.Scatter(
            x=corr_ic_df['Money IC'],
            y=slope * corr_ic_df['Money IC'] + intercept,
            mode="lines",
            marker_color=STANDARD_COLORS[2],
            # name=f"{block} - Regression",
            # marker=dict(color=COLORS[1]),
        ),
        row=1,
        col=1,
    )
    # Annotation
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        xanchor="right",
        yanchor="top",
        x=0.98,
        y=0.98,
        text=f"r={np.round(r_value, 3)}, p={np.round(p_value, 3)}",
        showarrow=False,
        row=1,
        col=1,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
    )

    ## Corr plot
    # Markers
    fig.add_trace(
        go.Scatter(
            x=corr_ic_df['Money Corr'],
            y=corr_ic_df['Other Corr'],
            mode="markers",
            name=model_name,
            hovertemplate=x_names,
            marker_color=STANDARD_COLORS[3],
        ),
        row=1,
        col=2,
    )
    # Trendline
    slope, intercept, r_value, p_value, std_err = linregress(corr_ic_df['Money Corr'], corr_ic_df['Other Corr'])
    fig.add_trace(
        go.Scatter(
            x=corr_ic_df['Money Corr'],
            y=slope * corr_ic_df['Money Corr'] + intercept,
            mode="lines",
            marker_color=STANDARD_COLORS[3],
            # name=f"{block} - Regression",
            # marker=dict(color=COLORS[1]),
        ),
        row=1,
        col=2,
    )
    # Annotation
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        xanchor="right",
        yanchor="top",
        x=0.98,
        y=0.98,
        text=f"r={np.round(r_value, 3)}, p={np.round(p_value, 3)}",
        showarrow=False,
        row=1,
        col=2,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
    )

    fig.update_layout(
        title=f"{batchfit.craving_models[model_name].craving_model_name} - Money vs Other condition",
        width=width,
        height=height,
        showlegend=False,
    )
    fig.layout["xaxis"]["title"]["text"] = "Money IC"
    fig.layout["yaxis"]["title"]["text"] = "Other IC"
    fig.layout["xaxis2"]["title"]["text"] = "Money Correlation"
    fig.layout["yaxis2"]["title"]["text"] = "Other Correlation"

    return fig


def corr_vs_ic(batchfit, model_name, metric="DIC", width=1000, height=500):
    corrs = _get_corrs(batchfit, model_name)
    ics = _get_ics(batchfit, model_name, metric)
    annotations = []
    corr_ic_df = pd.DataFrame.from_dict({
        "PID": batchfit.pid_subset,
        "Money Corr": corrs[:, 0],
        "Other Corr": corrs[:, 1],
        "Money IC": ics[0, :],
        "Other IC": ics[1, :],
    }).dropna()
    x_names = [f"{elem} - {np.argwhere(np.array(batchfit.pid_subset)==elem)[0][0]}" for elem in corr_ic_df['PID'].values]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=["Money", "Other"],
    )

    for i, block in enumerate(["Money", "Other"]):
        # PLOT POINTS
        fig.add_trace(
            go.Scatter(
                x=corr_ic_df[f'{block} Corr'],
                y=corr_ic_df[f'{block} IC'],
                mode="markers",
                name=block,
                hovertemplate=x_names,
                marker=dict(color=STANDARD_COLORS[i]),
            ),
            row=1,
            col=i + 1,
        )

        # PLOT REGRESSION LINE
        slope, intercept, r_value, p_value, std_err = linregress(corr_ic_df[f'{block} Corr'], corr_ic_df[f'{block} IC'])
        fig.add_trace(
            go.Scatter(
                x=corr_ic_df[f'{block} Corr'],
                y=slope * corr_ic_df[f'{block} Corr'] + intercept,
                mode="lines",
                name=f"{block} - Regression",
                marker=dict(color=STANDARD_COLORS[i]),
            ),
            row=1,
            col=i + 1,
        )

        # ANNOTATION OF R2
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="right",
            yanchor="top",
            x=0.98,
            y=0.98,
            text=f"r={np.round(r_value, 3)}, p={np.round(p_value, 3)}",
            showarrow=False,
            row=1,
            col=i + 1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
        )

    fig.update_layout(
        title=f"{batchfit.craving_models[model_name].craving_model_name} - Relationship between correlation and IC",
        xaxis_title=f"Money Correlation",
        yaxis_title=f"Other Correlation",
        width=width,
        height=height,
        showlegend=False,
    )
    fig.layout["xaxis"]["title"]["text"] = "Correlation"
    fig.layout["yaxis"]["title"]["text"] = "IC"
    fig.layout["xaxis2"]["title"]["text"] = "Correlation"
    fig.layout["yaxis2"]["title"]["text"] = "IC"

    return fig


def individual_parameter_estimation(
    batchfit, model_name, pid=None, pid_num=None, width=1000, height=500
):
    if pid is not None and pid_num is not None:
        raise ValueError("Only one of pid and pid_num can be specified")
    if pid is None and pid_num is None:
        raise ValueError("Either pid or pid_num must be specified")
    if pid_num is not None:
        pid = batchfit.pid_subset[pid_num]

    pid_df = _get_pid_params(batchfit, model_name, pid)

    return (
        px.box(
            pid_df,
            y=["w0", "w1", "w2"],
            color="block",
            width=800,
            height=500,
            title=f"PID: {pid}",
        ),
        pid_df,
    )


def group_parameter_estimation(batchfit, model_name, width=1000, height=500):

    group_estimates = pd.DataFrame(columns=["pid", "w0", "w1", "w2", "block"])
    for pid in batchfit.pid_subset:
        pid_df = _get_pid_params(batchfit, model_name, pid)
        if pid_df is None:
            continue
        w0_money = pid_df[pid_df["block"] == "money"]["w0"].mean()
        w1_money = pid_df[pid_df["block"] == "money"]["w1"].mean()
        w2_money = pid_df[pid_df["block"] == "money"]["w2"].mean()
        w0_other = pid_df[pid_df["block"] == "other"]["w0"].mean()
        w1_other = pid_df[pid_df["block"] == "other"]["w1"].mean()
        w2_other = pid_df[pid_df["block"] == "other"]["w2"].mean()

        group_estimates = pd.concat(
            [
                group_estimates,
                pd.DataFrame.from_dict(
                    {
                        "pid": [pid] * 2,
                        "w0": [w0_money, w0_other],
                        "w1": [w1_money, w1_other],
                        "w2": [w2_money, w2_other],
                        "block": ["money", "other"],
                    }
                ),
            ]
        )

    fig = go.Figure()
    money_w0 = group_estimates[group_estimates["block"] == "money"]["w0"]
    money_w1 = group_estimates[group_estimates["block"] == "money"]["w1"]
    money_w2 = group_estimates[group_estimates["block"] == "money"]["w2"]
    other_w0 = group_estimates[group_estimates["block"] == "other"]["w0"]
    other_w1 = group_estimates[group_estimates["block"] == "other"]["w1"]
    other_w2 = group_estimates[group_estimates["block"] == "other"]["w2"]
    fig.add_trace(
        go.Box(
            x=["w0"] * len(money_w0) + ["w1"] * len(money_w1) + ["w2"] * len(money_w2),
            y=money_w0.values.tolist()
            + money_w1.values.tolist()
            + money_w2.values.tolist(),
            name="money",
            boxpoints="all",
            pointpos=0,
        )
    )
    fig.add_trace(
        go.Box(
            x=["w0"] * len(other_w0) + ["w1"] * len(other_w1) + ["w2"] * len(other_w2),
            y=other_w0.values.tolist()
            + other_w1.values.tolist()
            + other_w2.values.tolist(),
            name="other",
            boxpoints="all",
            pointpos=0,
        )
    )
    fig.update_layout(
        title="Parameter Estimates",
        xaxis_title="Parameter",
        yaxis_title="Estimate",
        boxmode="group",
        width=800,
        height=500,
    )
    return fig, group_estimates

def group_blockdiff_parameter_estimation(batchfit, model_name, width=1000, height=500):

    group_estimates = pd.DataFrame(columns=["pid", "w0", "w1", "w2", "block"])
    for pid in batchfit.pid_subset:
        pid_df = _get_pid_params(batchfit, model_name, pid)
        if pid_df is None:
            continue
        w0_money = pid_df[pid_df["block"] == "money"]["w0"].mean()
        w1_money = pid_df[pid_df["block"] == "money"]["w1"].mean()
        w2_money = pid_df[pid_df["block"] == "money"]["w2"].mean()
        w0_other = pid_df[pid_df["block"] == "other"]["w0"].mean()
        w1_other = pid_df[pid_df["block"] == "other"]["w1"].mean()
        w2_other = pid_df[pid_df["block"] == "other"]["w2"].mean()

        group_estimates = pd.concat(
            [
                group_estimates,
                pd.DataFrame.from_dict(
                    {
                        "pid": [pid] * 2,
                        "w0": [w0_money, w0_other],
                        "w1": [w1_money, w1_other],
                        "w2": [w2_money, w2_other],
                        "block": ["money", "other"],
                    }
                ),
            ]
        )

    fig = go.Figure()
    money_w0 = group_estimates[group_estimates["block"] == "money"]["w0"]
    money_w1 = group_estimates[group_estimates["block"] == "money"]["w1"]
    money_w2 = group_estimates[group_estimates["block"] == "money"]["w2"]
    other_w0 = group_estimates[group_estimates["block"] == "other"]["w0"]
    other_w1 = group_estimates[group_estimates["block"] == "other"]["w1"]
    other_w2 = group_estimates[group_estimates["block"] == "other"]["w2"]

    diff_w0 = other_w0.values - money_w0.values
    diff_w1 = other_w1.values - money_w1.values
    diff_w2 = other_w2.values - money_w2.values

    fig.add_trace(
        go.Box(
            x=["w0"] * len(diff_w0) + ["w1"] * len(diff_w1) + ["w2"] * len(diff_w2),
            y=diff_w0.tolist()
            + diff_w1.tolist()
            + diff_w2.tolist(),
            name="Difference",
            boxpoints="all",
            pointpos=0,
        )
    )
    fig.update_layout(
        title="Parameter Estimates",
        xaxis_title="Parameter",
        yaxis_title="Estimate",
        # boxmode="group",
        width=width,
        height=height,
    )
    return fig, group_estimates


def trendlines(data_frame, x, y, color):

    figs = []
    # rp_vals = []
    for block in data_frame[color].unique():
        block_df = data_frame[data_frame[color] == block]
        slope, intercept, r_value, p_value, std_err = linregress(
            block_df[x], block_df[y]
        )
        # rp_vals.append([r_value, p_value])

        figs.append(
            go.Scatter(
                x=block_df[x],
                y=slope * block_df[x] + intercept,
                mode="lines",
                name=f"{block} - r={r_value:.2f}, p={p_value:.2f}",
            )
        )

    return figs  # , np.array(rp_vals)

