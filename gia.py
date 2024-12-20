#si quieres usar y entrenar este modelo , hacerlo correr en Google colab con un entorno
#Gpu 
#y descargar el modelo
# Línea 1 - Descargar librerías
import tensorflow as tf
import tensorflow_datasets as tfds
#---------------------------------------------
# Línea 2 - Descargar el set de datos Fashion MNIST
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
#---------------------------------------------
# Línea 3 - Ver metadatos del dataset
metadatos
#---------------------------------------------
# Línea 4 - Separar datos de entrenamiento y pruebas
datos_entrenamiento, datos_pruebas = datos['train'], datos['test']
#---------------------------------------------
# Línea 5 - Etiquetas de las categorías
nombres_clases = metadatos.features['label'].names
#---------------------------------------------
# Línea 6 - Ver las etiquetas
nombres_clases
#---------------------------------------------
# Línea 7 - Normalización de los datos
# Transformar valores de píxeles 0-255 a 0-1 para que la red entrene más rápido
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas

datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

# Guardar en caché para acelerar el entrenamiento
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()
#---------------------------------------------
# Línea 8 - Visualizar una imagen de entrenamiento
for imagen, etiqueta in datos_entrenamiento.take(1):
    break

imagen = imagen.numpy().reshape((28,28))
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
#---------------------------------------------
# Línea 9 - Dibujar más imágenes
plt.figure(figsize=(10,10))
for i, (imagen, etiqueta) in enumerate(datos_entrenamiento.take(25)):
    imagen = imagen.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.xlabel(nombres_clases[etiqueta])
plt.show()
#---------------------------------------------
# Línea 10 - Crear modelo con el InputLayer explícito
# Crear el modelo con la recomendación actualizada para InputLayer
from tensorflow.keras.layers import InputLayer

modelo = tf.keras.Sequential([
    InputLayer(shape=(28, 28, 1)),  # Forma de entrada explícita
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Para 10 clases
])
#---------------------------------------------
# Línea 11 - Compilar el modelo
modelo.compile(
    optimizer='adam',
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#---------------------------------------------
# Línea 12 - Obtener el número de ejemplos de entrenamiento y prueba
num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

print(num_ej_entrenamiento)
print(num_ej_pruebas)
#---------------------------------------------
# Línea 13 - Configurar tamaño de lote y mezclar datos
TAMANO_LOTE = 32
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)
#---------------------------------------------
# Línea 14 - Entrenar el modelo
import math
historial = modelo.fit(
    datos_entrenamiento,
    epochs=10,
    steps_per_epoch=math.ceil(num_ej_entrenamiento / TAMANO_LOTE)
)
#---------------------------------------------
# Línea 15 - Visualizar la pérdida del modelo durante el entrenamiento
plt.xlabel("# Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
#---------------------------------------------
# Línea 16 - Evaluar predicciones y visualizar resultados
import numpy as np
for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones = modelo.predict(imagenes_prueba)

def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    etiqueta_prediccion = np.argmax(arr_predicciones)
    color = 'blue' if etiqueta_prediccion == etiqueta_real else 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(
        nombres_clases[etiqueta_prediccion],
        100 * np.max(arr_predicciones),
        nombres_clases[etiqueta_real]), color=color)

def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0, 1])
    grafica[np.argmax(arr_predicciones)].set_color('red')
    grafica[etiqueta_real].set_color('blue')

filas = 5
columnas = 5
num_imagenes = filas * columnas
plt.figure(figsize=(2 * 2 * columnas, 2 * filas))
for i in range(num_imagenes):
    plt.subplot(filas, 2 * columnas, 2 * i + 1)
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    plt.subplot(filas, 2 * columnas, 2 * i + 2)
    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
#---------------------------------------------
# Línea 17 - Guardar el modelo en formato H5
modelo.save('modelo_exportado.h5')
# Guarda el modelo en formato TensorFlow.js
#---------------------------------------------
# Línea 18 - Instalar TensorFlow.js para conversión
!pip install --upgrade tensorflowjs
#---------------------------------------------
#linea 18,2
from tensorflow.keras.models import load_model

modelo = load_model("modelo_exportado.h5")
print(modelo.summary())
#---------------------------------------------
# Línea 19 - Convertir el modelo H5 a TensorFlow.js
!mkdir carpeta_salida
!tensorflowjs_converter --input_format keras modelo_exportado.h5 carpeta_salida
#---------------------------------------------
# Línea 20 - Verificar carpeta de salida
!ls carpeta_salida
#---------------------------------------------
#borrar
import tensorflow as tf
print(tf.__version__)
#---------------------------------------------
#---------------------------------------------
#---------------------------------------------





