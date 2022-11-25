import streamlit as st
import pathlib
from PIL import Image
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from yawbcc.demo import compute_grad_cam_heatmaps,color_segmentation,unet_segmentation
from yawbcc.datasets import load_wbc_dataset, WBCDataSequence
import tensorflow as tf
import cv2

# CONFIG PAGE
st.set_page_config(layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")
# CONFIG SIDEBAR


# DATA
classes = ['BASOPHIL', 'EOSINOPHIL', 'ERYTHROBLAST', 'IG', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL', 'PLATELET']
idx_to_cls = dict(enumerate(classes))
cls_to_idx = {c: i for i, c in idx_to_cls.items()}
rng = np.random.default_rng(seed=2022)
df = load_wbc_dataset('barcelona')
demodf = df.groupby('group').sample(n=10, random_state=rng.bit_generator).sort_index()
demods = WBCDataSequence(demodf['path'], demodf['group'].map(cls_to_idx), image_size=(256, 256))
images = np.concatenate([batch[0] for batch in demods])
DATA_DIR = pathlib.Path.home() / 'yawbcc_data'
# Load gradcam models
with tf.device('/CPU:0'):
    gc_conv = tf.keras.models.load_model(DATA_DIR / 'models' / 'gradcam_conv.hdf5')
    gc_clf = tf.keras.models.load_model(DATA_DIR / 'models' / 'gradcam_clf.hdf5')

# Load unet model
with tf.device('/CPU:0'):
    unet_cnn = tf.keras.models.load_model(DATA_DIR / 'models' / 'unet_256.hdf5')
idx = rng.choice(demodf.index)

# IMAGE SELECTION (Columns #1)
text_1,text_2,text_3,text_4 = st.columns([2,1,2,2])


text_3.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Séléction</p>", unsafe_allow_html=True)
text_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Aperçu de l'image</p>", unsafe_allow_html=True)


col_1,col_2,col_3,col_4,col_5,col_6 = st.columns([4,2,2,1,1,4])
col_3.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Importation</p>", unsafe_allow_html=True)
col_5.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Aléatoire</p>", unsafe_allow_html=True)
col_4.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Manuel</p>", unsafe_allow_html=True)
    # Bouton aléatoire
with col_5:
    alea_button = st.button("🎲")

with col_4:
    if alea_button:
        st.session_state.search_class = random.choice(demodf["group"].unique()) 
    class_selected = st.selectbox('CLASSE', demodf["group"].unique(),key='search_class')  
    if alea_button:
        st.session_state.search_image = random.choice(list(demodf[demodf["group"]==class_selected]["image"]))
    image_selected = st.selectbox('IMAGE',list(demodf[demodf["group"]==class_selected]["image"]),key='search_image')  
    image_name = image_selected   

with col_3:

    dd_img = st.file_uploader(label="",label_visibility="hidden")



with col_2:
    if dd_img:
        image_selected = dd_img
        image_test = Image.open(dd_img)  
        image_name = dd_img.name    
    else: 
        image_path = demodf[demodf["image"]==image_selected]["path"].item()
        image_test = Image.open(image_path) 
    st.image(image_test,caption=f"Image séléctionnée : {image_name}")
    
func_1,func_2,func_3 = st.columns([2,3,2])

func_2.markdown("<p style='padding: 20px; border: 2px solid white;text-align: center;font-size: 20px;'>Fonctions</p>", unsafe_allow_html=True)

but_1,but_2,but_3,but_4,but_5 = st.columns([5,1,1,1,5])
with but_2: 
    pred_button = st.button("Prediction")
with but_3:
    grad_button = st.button("Grad Cam")
with but_4:
    segm_button = st.button("Segmentation")

def highlight_matrix(x):
                max_color = f"background-color: rgba({', '.join(str(int(c*255)) for c in cmap(x.max()))});"
                low_color = 'color: rgba(255, 255, 255, 0.2);'
                condlist = [x <= 0.1, x == x.max()]
                choicelist = [low_color, max_color]
                return np.select(condlist, choicelist, default=None)

if pred_button:
    pred_1,pred_2,pred_3 = st.columns([1,2,1])
    with pred_2:
        with st.spinner(""):
            images = [image_path]
            files = [('images', (image, open(image, 'rb'))) for image in images]
            df_proba = pd.DataFrame(columns=[x.upper() for x in demodf["group"].unique()])
            cmap = plt.cm.Greens # Greens, Reds, Blues, jet, rainbow
            models_request = requests.get('https://yawbcc.demain.xyz/api/v1/models').json()
            for model in models_request:
                files = [('images', (image, open(image, 'rb'))) for image in images]
                predict_proba_link = f"https://yawbcc.demain.xyz/api/v1/models/{model}/predict_proba"
                proba_request = requests.post(predict_proba_link,files=files).json()
                df_temp = pd.DataFrame.from_dict(proba_request).rename({0:model},axis=0)
                df_proba = pd.concat([df_proba,df_temp])
        st.dataframe(df_proba.style.apply(highlight_matrix,axis=1),use_container_width=True)

elif grad_button:
    col_grad1,col_grad2,col_grad3,col_grad4,col_grad5 = st.columns([3,1,1,1,3])
    with col_grad2:
        with st.spinner(""):
            idx = demodf[demodf["image"]==image_name].index.to_list()[0]
            image = np.uint8(images[demodf.index.get_loc(idx)])
            heatmap = compute_grad_cam_heatmaps(image[None], gc_conv, gc_clf)[0].astype('uint8')
            cmap = plt.cm.get_cmap('jet')
            colors = np.uint8(255 * cmap(np.arange(256))[:, :3])
            colors[:30] = 0  # threshold low attention colors
            heatmap = cv2.resize(colors[heatmap], image.shape[:2])
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            gradcam = cv2.addWeighted(gray, 1, heatmap, 0.8, 0)
            st.image(image,caption="Image zoomée",use_column_width=True)
    with col_grad3:
        st.image(heatmap,caption="Heatmap générée",use_column_width=True)
    with col_grad4:
        st.image(gradcam,caption="Gradcam",use_column_width=True)

elif segm_button:
    idx = demodf[demodf["image"]==image_name].index.to_list()[0]
    image = np.uint8(images[demodf.index.get_loc(idx)])
    cmask = color_segmentation(image)
    umask = unet_segmentation(image, unet_cnn)

    cimg = cv2.bitwise_and(image, image, mask=cmask)
    uimg = cv2.bitwise_and(image, image, mask=umask)

    col_seg1,col_seg2,col_seg3,col_seg4 = st.columns([3,1,1,3])
    with col_seg2:
        st.image(cimg,caption="Segmentation avec couleurs",use_column_width=True)
    with col_seg3:
        st.image(uimg,caption="Segmentation avec UNet",use_column_width=True)
    



