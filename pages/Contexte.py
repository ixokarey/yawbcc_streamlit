import streamlit as st
import os
import pathlib
from PIL import Image
import numpy as np
import random

st.set_page_config(page_icon="🔬")
st.sidebar.markdown("Damien Corral damien.corral@gmail.com")
st.sidebar.markdown("Anastasiya Trushko Perney anastasia.trushko@gmail.com")
st.sidebar.markdown("Jordan Porcu jordan.porcu@gmail.com")
st.sidebar.markdown("Jérémy Lavergne jeremy.lav2009@gmail.com")
st.title("Project YAWBCC (Yet another white cell classification)")
st.header("Objectif")
st.write("L’objectif de ce projet est de classifier les cellules sanguines en fonction de leurs caractéristiques morphologiques en utilisant des techniques d’apprentissage profond (ou Deep Learning). Pour y parvenir, nous utiliserons un des réseaux neuronaux de convolution (CNN) avec une méthode d’apprentissage par transfert (Transfer Learning) ainsi qu’une optimisation des poids des modèles grâce à la méthode du Fine-tuning.")
    
st.header("Contexte")
st.image("cell_types.png")
st.write("""
    Les diagnostics de la majorité des maladies hématologiques commencent par une analyse morphologique des cellules sanguines périphériques. Ces dernières circulent dans les vaisseaux sanguins et contiennent trois types de cellules principales suspendues dans du plasma :
             
    - l’érythrocyte (globule rouge)
    - le leucocyte (globule blanc)
    - le thrombocyte (plaquette) 

    Les leucocytes sont les acteurs majeurs dans la défense de l’organisme contre les infections. Ce sont des cellules nucléées qui sont elle-mêmes divisées en trois classes :
    
    - les granulocytes (subdivisés en neutrophiles segmentés et en bande, en éosinophiles et en basophiles)
    - les lymphocytes
    - les monocytes

    Lorsqu’un patient est en bonne santé, la proportion des différents types de globules blancs dans le plasma est d’environ 54-62% pour les granulocytes, 25-33% pour les lymphocytes et 3-10% pour les monocytes. Cependant,
    en cas de maladie, par exemple une infection ou une anémie régénérative, cette proportion est modifiée en même temps que le nombre total de globules blancs, et on peut trouver des granulocytes immatures (IG) (promyélocytes, myélocytes et métamyélocytes) ou des précurseurs érythroïdes, comme les érythroblastes. 
    
    """)