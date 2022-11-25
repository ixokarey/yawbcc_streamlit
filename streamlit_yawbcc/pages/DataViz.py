import streamlit as st
import os
import pathlib
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_icon="📊")
st.sidebar.markdown("Damien Corral damien.corral@gmail.com")
st.sidebar.markdown("Anastasiya Trushko Perney anastasia.trushko@gmail.com")
st.sidebar.markdown("Jordan Porcu jordan.porcu@gmail.com")
st.sidebar.markdown("Jérémy Lavergne jeremy.lav2009@gmail.com")

df = pd.read_csv("predictions.csv")
st.title("Data Exploration and Vizualisation")
st.header("Origine de la donnée")
st.write("""  Nous avons mené notre projet en partant d’un jeu de données contenant un total de 17 092 images
    de cellules normales individuelles, qui ont été acquises à l’aide de l’analyseur CellaVision DM96 dans
    le laboratoire central de **la clinique hospitalière de Barcelone** (https://www.sciencedirect.com/science/article/pii/S2352340920303681).

    L’ensemble de données est organisé
    en huit groupes suivants : neutrophiles, éosinophiles, basophiles, lymphocytes, monocytes, granu-
    locytes immatures (promyélocytes, myélocytes et métamyélocytes), érythroblastes et plaquettes ou
    thrombocytes.""")

with st.expander("Extait du dataset"):
    st.dataframe(df.head())




with st.expander("Distribution des classes"):
    fig1 = plt.figure(figsize=(10, 15))
    plt.title("Avec 8 classes");
    sns.countplot(y=df.group)
    st.pyplot(fig1,clear_figure=True)

    fig2 = plt.figure(figsize=(10,15))
    plt.title("Avec 13 classes");
    sns.countplot(y=df['label'], hue=df['group'],dodge=False)
    st.pyplot(fig2)

st.markdown(""" Conclusions:
    - Dans le jeu de données de 17 092 images, il y a 8 classes: "basophile","neutrophile","ig","monocyte","éosinophile","érythroblaste","lymphocyte","plaquette".
    - La classe "neutrophile" comporte 3 types d'étiquettes : "BNE", "SNE" et "Neutrophile"
    - La classe "ig" comporte 4 types d'étiquettes : "MY", "PMY", "MMY", "IG".
    - Chaque autre classe n'a qu'un seul type d'étiquette.
    - Les images ont le même format.
    - Les images sont de tailles différentes :
         - 16 639 images d'une taille de 363x360 (hxw),
         - 250 images de 369x366,
         - 201 images de 360x360,
         - 2 images de 361x360, 
         - 1 image de 360x359, 
         - 1 image de 360x362, 
         - 1 image 360x361.""")