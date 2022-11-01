import streamlit as st
import pickle


def write():
    st.title('Training - Loss and Accuracy plot')
    with open("./model_data/MNIST_plot", 'rb') as f:
        plot_fig = pickle.load(f)

    st.plotly_chart(plot_fig)
    return
