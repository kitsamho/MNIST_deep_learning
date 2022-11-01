import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

@st.cache
def get_accuracy_and_loss_plot(running_loss_values, accuracy_values, title, y_left_title, y_right_title):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x_axis = [i for i in range(len(running_loss_values))]
    fig.add_trace(
        go.Scatter(x=x_axis, y=running_loss_values, name="yaxis data"),
        secondary_y=False)

    fig.add_trace(
        go.Scatter(x=x_axis, y=accuracy_values, name="yaxis2 data"),
        secondary_y=True)

    # Add figure title
    fig.update_layout(title_text=title)

    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")

    # Set y-axes titles
    fig.update_yaxes(title_text=f"<b>{y_left_title}</b>", secondary_y=False)
    fig.update_yaxes(title_text=f"<b>{y_right_title}</b>", secondary_y=True)

    return fig