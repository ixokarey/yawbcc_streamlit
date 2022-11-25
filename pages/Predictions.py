import streamlit as st
import os
import pathlib
from PIL import Image
import random
import pandas as pd

st.set_page_config(layout="centered",page_icon="🔮")
st.markdown("<h1 style='text-align: center; color: white;'>Predictions</h1>", unsafe_allow_html=True)
st.sidebar.markdown("Damien Corral damien.corral@gmail.com")
st.sidebar.markdown("Anastasiya Trushko Perney anastasia.trushko@gmail.com")
st.sidebar.markdown("Jordan Porcu jordan.porcu@gmail.com")
st.sidebar.markdown("Jérémy Lavergne jeremy.lav2009@gmail.com")

df = pd.read_csv("predictions.csv")
df = df[df["dataset"]!="munich"]

BASE_DIR = pathlib.Path.home () / 'yawbcc_data' / 'barcelona'

folder_dir = os.listdir(str(BASE_DIR))

button_cols = st.columns(8)
random_but = button_cols[4].button("Aléatoire")
manual_but = button_cols[3].button("Manuel")

if random_but:
    folder_selected = random.choice(folder_dir)
    img_dir = os.path.join(BASE_DIR,folder_selected)
    img_selection = df[df["group"].str.lower()==folder_selected]["image"].unique()
    img_selected = random.choice(img_selection)
    st.write("Image sélectionnée : " + img_selected)
    st.write("Classe : " + folder_selected.upper())

else: 
    folder_selected = st.selectbox('Selectionnez une classe', folder_dir)
    img_dir = os.path.join(BASE_DIR,folder_selected)
    #img_selection = os.listdir(img_dir) (10% du dataset)
    img_selection = df[df["group"].str.lower()==folder_selected]["image"].unique()
    img_selected = st.selectbox('Selectionnez une image', img_selection)

complete_img_dir = os.path.join(img_dir,img_selected)
selected_image = Image.open(complete_img_dir)

st.image(selected_image,use_column_width=True)

CLASS = folder_selected.upper()
IMAGE = img_selected

df_temp = df[(df["group"]==CLASS) & (df["image"]==IMAGE)].iloc[:,5:]
df_temp_corrige = df_temp.set_index('model').rename(columns=lambda x: x[:2])
classe_predite = df_temp.set_index('model').mean(axis=0).idxmax()

import matplotlib as plt
import numpy as np

cmap = plt.cm.Greens # Greens, Reds, Blues, jet, rainbow

def highlight_matrix(x):
    max_color = f"background-color: rgba({', '.join(str(int(c*255)) for c in cmap(x.max()))});"
    low_color = 'color: rgba(255, 255, 255, 0.2);'
    condlist = [x <= 0.1, x == x.max()]
    choicelist = [low_color, max_color]
    return np.select(condlist, choicelist, default=None)



#st.dataframe(df_temp_corrige.style.highlight_max(axis=1,color="green"),use_container_width=True)
#st.dataframe(df_temp_corrige.style.background_gradient(cmap ='Greens', vmin=0, vmax=1, axis=1).highlight_max(axis=1,color="green").set_properties(**{'color':'black'}),use_container_width=True)
st.dataframe(df_temp_corrige.style.apply(highlight_matrix, axis=1),use_container_width=True)
st.write("La classe prédite est : ", classe_predite)



