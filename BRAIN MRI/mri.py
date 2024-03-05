import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
from skimage import io
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
import os
import glob
import random
from google.colab import files #Librería para cargar ficheros directamente en Colab
# %matplotlib inline

# Necesitamos montar su disco usando los siguientes comandos:
# Para obtener más información sobre el montaje, puedes consultar: https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory
from google.colab import drive
drive.mount('/content/drive')

# Navegamos hasta el directorio My Drive para almacenar el conjunto de datos y herramientas.
# %cd /content/drive/My Drive/dataset/Healthcare AI Datasets/Brain_MRI

# Datos que contienen la ruta a Brain MRI y su mascara correspondiente
brain_df = pd.read_csv('data_mask.csv')

brain_df.info()

# Datos que contienen la ruta Brain MRI y su máscara con
brain_df = pd.read_csv('data_mask.csv')

brain_df.info()

brain_df.head(50)

brain_df.mask_path[1] # Ruta de la imagen de la MRI

brain_df.image_path[1] # Ruta de la máscara de segmentación

brain_df

brain_df['mask'].value_counts().index

# Usaremos plotly para hacer un diagrama de barras interactivo.
import plotly.graph_objects as go

fig = go.Figure([go.Bar(x = brain_df['mask'].value_counts().index, y = brain_df['mask'].value_counts())])
fig.update_traces(marker_color = 'rgb(0,200,0)', marker_line_color = 'rgb(0,255,0)', marker_line_width = 7, opacity = 0.6)
fig.show()

brain_df.mask_path

brain_df.image_path

plt.imshow(cv2.imread(brain_df.mask_path[623]))

plt.imshow(cv2.imread(brain_df.image_path[623]))

cv2.imread(brain_df.mask_path[623]).max()

cv2.imread(brain_df.mask_path[623]).min()

# Visualización básica, visualizaremos imágenes (MRI y Máscaras) en el dataset de forma separada
import random
fig, axs = plt.subplots(6,2, figsize=(16,32))
count = 0
for x in range(6):
  i = random.randint(0, len(brain_df)) # Seleccionamos un índice aleatorio
  axs[count][0].title.set_text("MRI del Cerebro") # Configuramos el título
  axs[count][0].imshow(cv2.imread(brain_df.image_path[i])) # Mostramos la MRI
  axs[count][1].title.set_text("Máscara - " + str(brain_df['mask'][i])) # Colocámos el título en la máscara (0 o 1)
  axs[count][1].imshow(cv2.imread(brain_df.mask_path[i])) # Mostramos la máscara correspondiente
  count += 1

fig.tight_layout()