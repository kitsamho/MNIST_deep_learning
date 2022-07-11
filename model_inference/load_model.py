import torch
import streamlit as st
from torchvision import transforms

@st.cache
def load_model(path: str):
    model = torch.load(path)
    return model


def get_inference_transformer():
    inference_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    return inference_transformer



