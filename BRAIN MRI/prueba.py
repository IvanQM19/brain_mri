import requests
import json

# Definimos la URL del endpoint
url = 'http://localhost:5000/visualize'

# Definimos el cuerpo de la solicitud
data = {'image_path': '/content/drive/My Drive/dataset/Healthcare AI Datasets/Brain_MRI'}

# Enviamos la solicitud POST
response = requests.post(url, json=data)

# Imprimimos la respuesta
print(response.json())
