import streamlit as st
import os
import pathlib
from PIL import Image
import numpy as np
import random
import pandas as pd

st.set_page_config(page_icon="🪛")
st.markdown("<h1 style='text-align: center; color: white;'>Modelisation</h1>", unsafe_allow_html=True)
st.sidebar.markdown("Damien Corral damien.corral@gmail.com")
st.sidebar.markdown("Anastasiya Trushko Perney anastasia.trushko@gmail.com")
st.sidebar.markdown("Jordan Porcu jordan.porcu@gmail.com")
st.sidebar.markdown("Jérémy Lavergne jeremy.lav2009@gmail.com")

model_list = ["VGG16","ResNet50","MobileNet","Xception"]

selected_model = st.selectbox("Choisissez un modèle : ", model_list)
st.title(selected_model)
BASE_DIR = selected_model

with st.expander("Architecture"):
    cols = st.columns(2)
    cols[0].image(BASE_DIR+"/summary_initial.png",caption="Sans fine-tuning")
    cols[1].image(BASE_DIR+"/summary_ft.png",caption="Avec fine-tuning")

    if selected_model == "VGG16":
        centering_cols = st.columns(3)
        centering_cols[1].image(BASE_DIR+"/arch.png",caption= "Architecture du modèle " + str(selected_model))
    else:
        st.image(BASE_DIR+"/arch.png",caption= "Architecture du modèle " + str(selected_model))

with st.expander("Entraînement"):
    st.image(BASE_DIR+"/fit_1_10.png",caption="Sans fine-tuning")
    st.image(BASE_DIR+"/fit_11_20.png",caption="Avec fine-tuning")

with st.expander("Résultats"):
    st.image(BASE_DIR+"/acc_loss_plot.png",caption = "Courbe d'entraînement (Loss/Accuracy)")
    result_col = st.columns(2)
    result_col[0].image(BASE_DIR+"/cr.png",caption = "Rapport de classification")
    result_col[1].image(BASE_DIR+"/cm_heatmap.png", caption="Matrice de confusion")
