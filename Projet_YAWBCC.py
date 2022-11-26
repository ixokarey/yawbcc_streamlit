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

about = {"About":"Anastasiya Trushko Perney anastasia.trushko@gmail.com, Jérémy Lavergne jeremy.lav2009@gmail.com, Damien Corral damien.corral@gmail.com, Jordan Porcu jordan.porcu@gmail.com"}

st.set_page_config(page_icon="🩸",menu_items=about)

pages = st.source_util.get_pages('main.py')
new_page_names = {
  'Projet YAWBCC': '🩸 Projet YAWBCC',
  'Contexte' : '🔬 Contexte',
  'DataViz' : '📊 Visualisation des données',
  'Modelisation': '📈 Modelisation',
  'Predictions' : '🔮 Predictions',
  'Preprocessing' : '🖼️ Preprocessing'
}