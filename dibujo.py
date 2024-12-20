from tkinter import Tk, Canvas, Button, Label
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Cargar el modelo preentrenado
model = tf.keras.models.load_model("fashion_mnist_cnn_model.h5")

# Etiquetas en español
clothing_items = [
    "Camiseta", "Pantalón", "Jersey", "Vestido", "Abrigo",
    "Sandalias", "Camisa", "Zapatillas", "Bolso", "Botines"
]

# Configuración de Tkinter
root = Tk()
root.title("Predicción de Ropa")
canvas = Canvas(root, width=224, height=224, bg="white")
canvas.pack()

x1, y1 = None, None  # Inicializar coordenadas

# Función para dibujar círculos pequeños sobre el canvas
def draw(event):
    global x1, y1
    x2, y2 = event.x, event.y
    radius = 9  # Radio del círculo (ajustar tamaño según preferencia)

    if x1 is not None and y1 is not None:
        # Dibujar un pequeño círculo en la nueva posición
        canvas.create_oval(x2 - radius, y2 - radius, x2 + radius, y2 + radius, fill="black", outline="black")

    x1, y1 = x2, y2  # Actualizar coordenadas para el próximo punto

# Función para limpiar el canvas
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="Predicción: ")

# Función para predecir la categoría de la imagen
def predict():
    # Crear una nueva imagen en blanco de 224x224
    img = Image.new("L", (224, 224), 255)
    draw = ImageDraw.Draw(img)

    # Dibujar sobre la imagen desde el canvas
    for item in canvas.find_all():
        coords = canvas.coords(item)
        # Dibujar los círculos pequeños sobre la imagen de 224x224
        draw.ellipse([coords[0] - 3, coords[1] - 3, coords[0] + 3, coords[1] + 3], fill=0)  # Ajustar tamaño del círculo

    # Redimensionar la imagen a 28x28 y normalizar
    img = img.resize((28, 28))  # Cambiar tamaño a 28x28
    img_array = np.array(img) / 255.0  # Normalizar la imagen a rango [0, 1]
    img_array = 1 - img_array  # Invertir los colores (de blanco a negro y viceversa)
    img_array = img_array.reshape(1, 28, 28, 1)  # Añadir la dimensión necesaria para la predicción

    # Realizar la predicción
    prediction = model.predict(img_array)
    label = np.argmax(prediction)

    # Mostrar la predicción
    result_label.config(text=f"Predicción: {clothing_items[label]}")

# Conectar el evento de dibujar al canvas
canvas.bind("<B1-Motion>", draw)

# Botones y etiquetas
predict_button = Button(root, text="Predecir", command=predict)
predict_button.pack()

clear_button = Button(root, text="Limpiar", command=clear_canvas)
clear_button.pack()

result_label = Label(root, text="Predicción: ")
result_label.pack()

root.mainloop()
