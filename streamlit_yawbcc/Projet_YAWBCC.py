import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'<streamlit>'])
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import requests
from PIL import Image
import os
import urllib.request
import pathlib
import shutil

st.set_page_config(page_icon="🩸",menu_items={"About":"Test"})
st.sidebar.markdown("Damien Corral damien.corral@gmail.com")
st.sidebar.markdown("Anastasiya Trushko Perney anastasia.trushko@gmail.com")
st.sidebar.markdown("Jordan Porcu jordan.porcu@gmail.com")
st.sidebar.markdown("Jérémy Lavergne jeremy.lav2009@gmail.com")
BASE_DIR = pathlib.Path.home () / 'yawbcc_data'

pages = st.source_util.get_pages('main.py')
new_page_names = {
  'Projet YAWBCC': '🩸 Projet YAWBCC',
  'Contexte' : '🔬 Contexte',
  'DataViz' : '📊 Visualisation des données',
  'Modelisation': '📈 Modelisation',
  'Predictions' : '🔮 Predictions',
  'Preprocessing' : '🖼️ Preprocessing'
}

for key, page in pages.items():
  if page['page_name'] in new_page_names:
    page['page_name'] = new_page_names[page['page_name']]


def download_dataset():
    DATASETS = {
        'barcelona': 'https://cloud.minesparis.psl.eu/index.php/s/r9oFCMOTI5zVcd9/download',
        'barcelona_remapped': 'https://cloud.minesparis.psl.eu/index.php/s/MmAkMdd9wrmcgUq/download',
        'barcelona_256_masked': 'https://cloud.minesparis.psl.eu/index.php/s/S5mPzVRXmwXUiYc/download',
    }
    for dataset in DATASETS.keys():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR = BASE_DIR / dataset
        arch, _ = urllib.request.urlretrieve(DATASETS[dataset])
        shutil.unpack_archive(arch, BASE_DIR, 'zip')

def download_images():
    URL = {"images" : "https://drive.google.com/uc?export=download&id=1h3aLDq31HDG4QWcJXrPP2bzZs5RUlC1Y"}
    for dataset in URL.keys():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR = BASE_DIR / dataset
        arch, _ = urllib.request.urlretrieve(URL[dataset])
        shutil.unpack_archive(arch, BASE_DIR, 'zip')



dow_ds = st.button("Télécharger datasets")
dow_img = st.button("Télécharger images")

if dow_ds:
    download_dataset()

if dow_img:
    download_images()