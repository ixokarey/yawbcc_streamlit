import streamlit as st
import os
import pathlib
from PIL import Image
import numpy as np
import random

# Configuration de la page
# Page setting
st.set_page_config(layout="wide")

# Importation de style.css
# style.css import
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")

# Titre 
# Title
title_1,title_2,title_3 = st.columns([4,6,4])
title_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Pré-processing</p>", unsafe_allow_html=True)

sub_1,sub_2,sub_3,sub_4 = st.columns([4,3,3,4])
sub_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Ségmentation par intensité de couleur</p>", unsafe_allow_html=True)
sub_3.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Ségmentation par UNet</p>", unsafe_allow_html=True)

img1 = Image.open(r"images\segm_couleur.png")
sub_2.image(img1,use_column_width=True)

img2 = Image.open(r"images\unet_architecture.png")
sub_3.image(img2,use_column_width=True)

img3 = Image.open(r"images\segm_unet.png")
sub_3.image(img3,use_column_width=True)   