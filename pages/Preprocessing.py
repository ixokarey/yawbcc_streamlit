import streamlit as st
import os
import pathlib
from PIL import Image
import numpy as np
import random

st.set_page_config(page_icon="🖼️")
st.markdown("<h1 style='text-align: center; color: white;'>Preprocessing</h1>", unsafe_allow_html=True)
st.sidebar.markdown("Damien Corral damien.corral@gmail.com")
st.sidebar.markdown("Anastasiya Trushko Perney anastasia.trushko@gmail.com")
st.sidebar.markdown("Jordan Porcu jordan.porcu@gmail.com")
st.sidebar.markdown("Jérémy Lavergne jeremy.lav2009@gmail.com")

BASE_DIR = pathlib.Path.home () / 'yawbcc_data' / 'barcelona_256_masked'

folder_dir = os.listdir(str(BASE_DIR))

button_cols1 = st.columns(8)
random_but1 = button_cols1[4].button("Aléatoire")
manual_but1 = button_cols1[3].button("Manuel")

if random_but1:
    folder_selected = random.choice(folder_dir)
    img_dir = os.path.join(BASE_DIR,folder_selected)
    img_selection = os.listdir(img_dir)
    img_selected = random.choice(img_selection)
    st.write("Image sélectionnée : " + img_selected)
    st.write("Classe : " + folder_selected.upper())

else: 
    folder_selected = st.selectbox('Selectionnez une classe', folder_dir)
    img_dir = os.path.join(BASE_DIR,folder_selected)
    img_selection = os.listdir(img_dir)
    img_selected = st.selectbox('Selectionnez une image', img_selection)






complete_img_dir = os.path.join(img_dir,img_selected)
selected_image = Image.open(complete_img_dir)

REMAPPED_DIR = str(pathlib.Path.home () / 'yawbcc_data' / 'barcelona_remapped') + "/" + folder_selected + "/" + img_selected
remapped_img = Image.open(REMAPPED_DIR)

ORIGINAL_DIR = str(pathlib.Path.home () / 'yawbcc_data' / 'barcelona') + "/" + folder_selected + "/" + img_selected
original_img = Image.open(ORIGINAL_DIR)

random_but=False



list = [original_img,remapped_img,selected_image]
cols_image = st.columns(3) 
cols_image[0].image(list[0],caption = "Originale",use_column_width=True)
cols_image[1].image(list[1],caption = "Remappée",use_column_width=True)
cols_image[2].image(list[2],caption = "Segmentée",use_column_width=True)

