# neural network
import torch
import torchvision

# custom dependancies
from load_model import load_model, get_inference_transformer
from images import load_image_from_canvas

# data and plotting
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# streamlit
import streamlit as st
from streamlit_drawable_canvas import st_canvas


def get_inference(image_path: str, inference_transformer: torchvision.transforms, model: torch.nn):
    image_canvas_transformed = inference_transformer(image_path)
    img = image_canvas_transformed.view(1, 784).float()
    model.eval()
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probabilities = list(ps.numpy()[0])
    print("Predicted Digit =", probabilities.index(max(probabilities)))
    return probabilities


st.title('MNIST Digit Predictor')
# Create a canvas component
canvas = st_canvas(
    fill_color="#FFFFFF",
    stroke_width=11,
    background_color="#000000",
    stroke_color='#FFFFFF',
    background_image=None,
    update_streamlit=True,
    height=28 * 12,
    width=28 * 12,
    drawing_mode='freedraw',
    point_display_radius=0,
    key="canvas")

# save image locally
plt.imsave('image_to_predict.jpeg', canvas.image_data)

# load image
image = load_image_from_canvas('image_to_predict.jpeg')

# load the trained model
model = load_model("../model_data/MNIST_model.pt")

# get inference transformer
inference_transformer = get_inference_transformer()

# get inference for new image
predict_prob = get_inference(image, inference_transformer, model)

# plot results
fig = px.bar(pd.DataFrame(predict_prob))
fig = fig.update_xaxes(title='Number Predicted', range=[0, 10])
fig = fig.update_yaxes(title='Probability', range=[0, 1])
st.plotly_chart(fig)
