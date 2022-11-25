import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib
from PIL import Image
import numpy as np
import random
from sklearn.metrics import classification_report

df=pd.read_csv('df.csv', index_col=0) 

st.sidebar.title("Navigation")
pages = ["Project Presentation", "Data Exploration", "Preprocessing","CNN Models", "Predictions"]
page=st.sidebar.radio("", pages)

st.sidebar.title("Authors:")
    
st.sidebar.markdown("Damien Corral damien.corral@gmail.com")
st.sidebar.markdown("Anastasiya Trushko Perney anastasia.trushko@gmail.com")
st.sidebar.markdown("Jordan Porcu jordan.porcu@gmail.com")
st.sidebar.markdown("Jérémy Lavergne jeremy.lav2009@gmail.com")


if page == pages[0]:
    st.title("Project YAWBCC (Yet another white cell classification)")
    st.header("Objectif")
    st.write("""L’objectif de ce projet est de classifier les cellules sanguines en fonction de leurs caractéristiques 
    morphologiques en utilisant des techniques d’apprentissage profond (ou Deep Learning). Pour y parvenir,
     nous utiliserons un des réseaux neuronaux de convolution (CNN) avec une méthode d’apprentissage par transfert (Transfer Learning)
      ainsi qu’une optimisation des poids des modèles grâce à la méthode du Fine-tuning.""")
    
    st.header("Contexte")
    st.image("cell_types.png")
    st.write("""
    Les diagnostics de la majorité des maladies hématologiques commencent par une analyse morphologique des cellules sanguines périphériques.
     Ces dernières circulent dans les vaisseaux sanguins et contiennent trois types de cellules principales suspendues dans du plasma :
             
    - l’érythrocyte (globule rouge)
    - le leucocyte (globule blanc)
    - le thrombocyte (plaquette) 

    Les leucocytes sont les acteurs majeurs dans la défense de l’organisme contre les infections. Ce sont des cellules nucléées qui sont elle-mêmes divisées en trois classes :
    
    - les granulocytes (subdivisés en neutrophiles segmentés et en bande, en éosinophiles et en basophiles)
    - les lymphocytes
    - les monocytes

    Lorsqu’un patient est en bonne santé, la proportion des différents types de globules blancs dans le plasma est
     d’environ 54-62% pour les granulocytes, 25-33% pour les lymphocytes et 3-10% pour les monocytes. Cependant,
    en cas de maladie, par exemple une infection ou une anémie régénérative, cette proportion est modifiée en même
     temps que le nombre total de globules blancs, et on peut trouver des granulocytes immatures (IG)
      (promyélocytes, myélocytes et métamyélocytes) ou des précurseurs érythroïdes, comme les érythroblastes. 
    
    """)

if page == pages[1]:

    st.title("Data Exploration and Vizualisation")
    st.header("Origine de la donnée")
    st.write("""  Nous avons mené notre projet en partant d’un jeu de données contenant un total de 17 092 images
    de cellules normales individuelles, qui ont été acquises à l’aide de l’analyseur CellaVision DM96 dans
    le laboratoire central de **la clinique hospitalière de Barcelone** (https://www.sciencedirect.com/science/article/pii/S2352340920303681).

    L’ensemble de données est organisé
    en huit groupes suivants : neutrophiles, éosinophiles, basophiles, lymphocytes, monocytes, granu-
    locytes immatures (promyélocytes, myélocytes et métamyélocytes), érythroblastes et plaquettes ou
    thrombocytes.""")

    option_exp = ["Data Frame","Les images de cellules correspondant à chacun des 8 classes","Répartition des images par classe et par label","Balance des couleurs", "Analyse des images selon leur teille" ]
    exp_data = st.selectbox(label='Choose from the List of Data Exploration', options=option_exp)
    
    if exp_data == "Data Frame":    
        st.header('DataFrame')
        st.dataframe(df.head())
    
    if exp_data == "Les images de cellules correspondant à chacun des 8 classes":
        st.header('Visualisation of our dataset')
        st.image(image='images_group.png',caption='Cell Images corresponding to each cell class.')

    if exp_data == "Répartition des images par classes et par label":  
        st.header('Dataset Exploration')
        fig, axes = plt.subplots(2, 1, figsize=(15, 20))
        plt.rcParams['font.size'] = '20'
        sns.countplot(ax=axes[0], y=df['group'])
        sns.countplot(ax=axes[1], y=df['label'], hue=df['group'],dodge=False)
        st.pyplot(fig)

    if exp_data == "Balance des couleurs":  
        st.image('ig.png')   
        st.image('color_hist_ig.png')
        st.markdown("""
        Dans l'échantillon précédent, nous constatons un problème au niveau des couleurs des images.
        Bien que les images des cellules aient été prises avec le même équipement, le traitement photographique
        est différent selon certaines classes (vérifié lors de l'extraction des données EXIF des images).
        """)

    if exp_data == "Analyse des images selon leur teille": 
        st.header('Analyse des images selon leur teille') 
        st.markdown(""" 


        | Width | Height | Count|
        |-----|------|-----|
        | 359 | 360 | 1 |
        | 360 |363| 16639 |
        | 360 | 360 | 198| 
        | 360 | 361 | 2 |
        | 361 | 360 | 1 |
        | 362 | 360 | 1 |
        | 366 | 369 | 250 | 
        """)
    
    
    st.header('Conclusions:')
    st.markdown(""" 
    - Dans le jeu de données de 17 092 images, il y a 8 classes:
        - basophile,
        - neutrophile,
        - ig,
        - monocyte,
        - éosinophile,
        - érythroblaste,
        - lymphocyte,
        - plaquette.
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

if page == pages[2]: 
    st.markdown("<h1 style='text-align: center; color: white;'>Preprocessing</h1>", unsafe_allow_html=True)

    BASE_DIR = pathlib.Path.home () / 'yawbcc_data' / 'barcelona_256_masked'

    folder_dir = os.listdir(str(BASE_DIR))

    button_cols = st.columns(8)
    random_but = button_cols[4].button("Aléatoire")
    manual_but = button_cols[3].button("Manuel")

    if random_but:
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
    cols = st.columns(3) 
    cols[0].image(list[0],caption = "Originale",use_column_width=True)
    cols[1].image(list[1],caption = "Remappée",use_column_width=True)
    cols[2].image(list[2],caption = "Segmentée",use_column_width=True)  

if page == pages[3]:
    models = ['VGG16','MobileNetV2','Xception','ResNet50']
    st.header('Select a model from the list')
    options=st.selectbox(label='Select from the list of models', options=models)
    if options=='VGG16':

        st.header(f"Les courbes d'apprentissage du modèle {models[0]}")
        st.image('acc_loss_plot_VGG16.png')

        df2=pd.read_csv('predictions.csv')
        df3=df2.loc[df2.dataset=='barcelona']

        df_vgg16_ft=df3.loc[df3.model=='vgg16_ft']
        df_vgg16_tl=df3.loc[df3.model=='vgg16_tl']

        res1 = df_vgg16_tl.iloc[:,6:14]
        res1['prediction'] = res1.idxmax(axis=1)

        res2 = df_vgg16_ft.iloc[:,6:14]
        res2['prediction'] = res2.idxmax(axis=1)

        cm1 = pd.crosstab(df_vgg16_tl['group'], res1['prediction'], rownames=['Real'], colnames=['Pred'])
        cm2 = pd.crosstab(df_vgg16_ft['group'], res2['prediction'], rownames=['Real'], colnames=['Pred'])

        st.header('Confusion matrix')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Transfer Learning")
            fig1=plt.figure()
            ax1 = sns.heatmap(cm1, cmap='viridis', annot=True, fmt='01')
            st.pyplot(fig1)

        with col2:
            st.subheader("After Fine-Tuning")
            fig2=plt.figure()
            ax2 = sns.heatmap(cm2, cmap='viridis', annot=True, fmt='01')
            st.pyplot(fig2)

        st.header('Classification report')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transfer Learning")
            cr1=classification_report(df_vgg16_ft['group'], res1['prediction'], output_dict=True)
            cr1 = pd.DataFrame(cr1).transpose()
            st.dataframe(cr1)

        with col2:
            st.subheader("Fine-Tuning")
            cr2=classification_report(df_vgg16_ft['group'], res2['prediction'], output_dict=True)
            cr2 = pd.DataFrame(cr2).transpose()
            st.dataframe(cr2)

    if options=='Xception':

        st.header(f"Les courbes d'apprentissage du modèle {models[2]}")
        st.image('acc_loss_plot_Xception.png')

        df2=pd.read_csv('predictions.csv')
        df3=df2.loc[df2.dataset=='barcelona']

        df_xception_ft=df3.loc[df3.model=='xception_ft']
        df_xception_tl=df3.loc[df3.model=='xception_tl']

        res1 = df_xception_tl.iloc[:,6:14]
        res1['prediction'] = res1.idxmax(axis=1)

        res2 = df_xception_ft.iloc[:,6:14]
        res2['prediction'] = res2.idxmax(axis=1)

        cm1 = pd.crosstab(df_xception_tl['group'], res1['prediction'], rownames=['Real'], colnames=['Pred'])
        cm2 = pd.crosstab(df_xception_ft['group'], res2['prediction'], rownames=['Real'], colnames=['Pred'])

        st.header('Confusion matrix')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Transfer Learning")
            fig1=plt.figure()
            ax1 = sns.heatmap(cm1, cmap='viridis', annot=True, fmt='01')
            st.pyplot(fig1)

        with col2:
            st.subheader("After Fine-Tuning")
            fig2=plt.figure()
            ax2 = sns.heatmap(cm2, cmap='viridis', annot=True, fmt='01')
            st.pyplot(fig2)
        
        

        st.header('Classification report')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Transfer Learning")
            cr1=classification_report(df_xception_tl['group'], res1['prediction'], output_dict=True)
            cr1 = pd.DataFrame(cr1).transpose()
            st.dataframe(cr1)

        with col2:
            st.subheader("After Fine-Tuning")
            cr2=classification_report(df_xception_ft['group'], res2['prediction'], output_dict=True)
            cr2 = pd.DataFrame(cr2).transpose()
            st.dataframe(cr2)            

    if options=='ResNet50':
        st.header(f"Les courbes d'apprentissage du modèle {models[0]}")
        st.image('acc_loss_plot_ResNet50V2.png')

        df2=pd.read_csv('predictions.csv')
        df3=df2.loc[df2.dataset=='barcelona']

        df_resnet50v2_ft=df3.loc[df3.model=='resnet50v2_ft']
        df_resnet50v2_tl=df3.loc[df3.model=='resnet50v2_tl']

        res1 = df_resnet50v2_tl.iloc[:,6:14]
        res1['prediction'] = res1.idxmax(axis=1)

        res2 = df_resnet50v2_ft.iloc[:,6:14]
        res2['prediction'] = res2.idxmax(axis=1)

        cm1 = pd.crosstab(df_resnet50v2_tl['group'], res1['prediction'], rownames=['Real'], colnames=['Pred'])
        cm2 = pd.crosstab(df_resnet50v2_ft['group'], res2['prediction'], rownames=['Real'], colnames=['Pred'])

        st.header('Confusion matrix')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Transfer Learning")
            fig1=plt.figure()
            ax1 = sns.heatmap(cm1, cmap='viridis', annot=True, fmt='01')
            st.pyplot(fig1)

        with col2:
            st.subheader("After Fine-Tuning")
            fig2=plt.figure()
            ax2 = sns.heatmap(cm2, cmap='viridis', annot=True, fmt='01')
            st.pyplot(fig2)

        st.header('Classification report')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transfer Learning")
            cr1=classification_report(df_resnet50v2_ft['group'], res1['prediction'], output_dict=True)
            cr1 = pd.DataFrame(cr1).transpose()
            st.dataframe(cr1)

        with col2:
            st.subheader("Fine-Tuning")
            cr2=classification_report(df_resnet50v2_ft['group'], res2['prediction'], output_dict=True)
            cr2 = pd.DataFrame(cr2).transpose()
            st.dataframe(cr2)
    

if page == pages[4]:

    #st.set_page_config(layout="centered")
    st.markdown("<h1 style='text-align: center; color: white;'>Predictions</h1>", unsafe_allow_html=True)

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

    st.image(selected_image, use_column_width=True)

    CLASS = folder_selected.upper()
    IMAGE = img_selected

   # headers_props = [('text-align','center'),('font-size','1.1em')]
    #styles = dict(selector='th.col_heading.level0',props=headers_props)


    df_temp = df[(df["group"]==CLASS) & (df["image"]==IMAGE)].iloc[:,5:]
    df_temp_corrige = df_temp.set_index('model').rename(columns=lambda x: x[:2])
    classe_predite = df_temp.set_index('model').mean(axis=0).idxmax()

    #st.dataframe(df_temp_corrige.style.highlight_max(axis=1,color="green"),use_container_width=True)
    st.dataframe(df_temp_corrige.style.background_gradient(cmap ='Greens', vmin=0, vmax=1, axis=1).highlight_max(axis=1,color="green").set_properties(**{'color':'black'}),use_container_width=True)

    st.write("La classe prédite est : ", classe_predite) 
        