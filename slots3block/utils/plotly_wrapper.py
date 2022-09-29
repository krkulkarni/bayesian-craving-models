# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import pandas as pd


# def plot_box(x, y, data=None, adjust_size=(500, 500)):

#     fig = go.Figure()

#     if data:
#         x_data = data[x]
#         y_data = data[y]
#     else:
#         x_data = x
#         y_data = y

#     fig.add_trace(go.Box(y=y_data, x=x_data, boxpoints="all", jitter=0.3, pointpos=-0))

#     height, width = adjust_size
#     fig.update_layout(autosize=False, width=width, height=height)

#     return fig


def plot_scatter(
    x,
    y,
    data=None,
    regplot=None,
    x_var=None,
    y_var=None,
    color=None,
    adjust_size=(500, 500),
):

    if data:
        fig = px.scatter(
            data_frame=data,
            x=x,
            y=y,
            color=color,
            height=adjust_size[0],
            width=adjust_size[1],
            trendline=regplot,
        )
    else:
        if x_var is None:
            x_var = "var1"
        if y_var is None:
            y_var = "var2"
        df_dict = {
            x_var: x,
            y_var: y,
        }
        if color:
            df_dict["color"] = color

        data = pd.DataFrame.from_dict(df_dict)
        fig = px.scatter(
            data_frame=data,
            x=x_var,
            y=y_var,
            color=color,
            height=adjust_size[0],
            width=adjust_size[1],
            trendline=regplot,
        )

    return fig
