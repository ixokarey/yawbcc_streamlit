import streamlit as st
from PIL import Image

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

# Sous titre 1
# Subtitle #1


select_preprocessing = title_2.selectbox("Sélectionnez la méthode de ségmentation",["Ségmentation par computer vision","Ségmentation par deep learning"])

if select_preprocessing == "Ségmentation par computer vision":
    title_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Ségmentation par intensité de couleur</p>", unsafe_allow_html=True)
    title_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Méthode</p>", unsafe_allow_html=True)
    title_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Résultat</p>", unsafe_allow_html=True)
    with Image.open("images/segm_couleur.png") as img:
        title_2.image(img,use_column_width=True)

elif select_preprocessing == "Ségmentation par deep learning":
    title_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Ségmentation par U-Net</p>", unsafe_allow_html=True)
    title_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Méthode</p>", unsafe_allow_html=True)
    title_2.write("Utilisation du réseau U-Net pour générer un masque")
    title_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Résultat</p>", unsafe_allow_html=True)
    with Image.open("images/unet_architecture.png") as img:
        title_2.image(img,use_column_width=True)
    with Image.open("images/segm_unet.png") as img:
        title_2.image(img,use_column_width=True)   
