from flask import Flask, send_file, request
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/visualize', methods=['POST'])
def visualize_image():
    data = request.json
    image_path = data['image_path']
    
    # Cargamos la imagen con OpenCV
    img = cv2.imread(image_path)
    
    # Convertimos la imagen a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Creamos una figura y un eje con matplotlib
    fig, ax = plt.subplots()
    
    # Mostramos la imagen en el eje
    ax.imshow(img)
    
    # Creamos un buffer en memoria
    buf = io.BytesIO()
    
    # Guardamos la figura en el buffer
    plt.savefig(buf, format='png')
    
    # Cerramos la figura para liberar memoria
    plt.close(fig)
    
    # Codificamos la imagen en base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    buf.close()
    
    return {'image': image_base64}

if __name__ == '__main__':
    app.run(debug=True)
