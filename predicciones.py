import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Ruta del modelo y carpeta de test
model_path = 'best_model_efficientnetb4.h5'  # Ruta de tu modelo entrenado
test_dir = 'Train'  # Ruta de la carpeta de imágenes para test
csv_path = 'data_refinado.csv'  # Ruta del archivo CSV con las etiquetas

# Cargar el modelo entrenado
model = load_model(model_path, compile=False)  # No compilar para evitar errores de versiones de TensorFlow

# Leer el archivo CSV y crear un diccionario de etiquetas
df = pd.read_csv(csv_path)
labels_dict = dict(zip(df['image'], df['label']))

# Función para preprocesar la imagen
IMG_SIZE = (380, 380)  # Tamaño recomendado para EfficientNetB4

@tf.function
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)  # Asegúrate de usar el mismo preprocesamiento que en el entrenamiento
    return img

# Función para mostrar imagen con su clasificación
def show_classification(image_path, predicted_class, real_label):
    img = plt.imread(image_path)
    plt.imshow(img)
    pred_label = 'Real' if predicted_class == 1 else 'Falsa'
    real_label_text = 'Real' if real_label == 1 else 'Falsa'
    plt.title(f"Predicción: {pred_label} (Esperado: {real_label_text})")
    plt.axis('off')  # Para no mostrar los ejes
    plt.show()

# Contadores para las predicciones correctas e incorrectas
correct_predictions = 0
incorrect_predictions = 0

# Iterar sobre las imágenes en la carpeta de test
for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)
    
    # Preprocesar la imagen
    img = preprocess_image(image_path)
    img = tf.expand_dims(img, axis=0)  # Agregar la dimensión del batch (1, 224, 224, 3)
    
    # Realizar la predicción
    prediction = model.predict(img)
    predicted_class = (prediction > 0.5).astype("int32")[0][0]  # Binario, si es mayor que 0.5 es "Real" (1), de lo contrario "Falsa" (0)

    # Obtener la etiqueta real desde el diccionario
    real_label = labels_dict.get(image_name, 0)  # Asumimos 0 (Falsa) si no se encuentra la etiqueta

    # Imprimir la evaluación de la predicción
    if predicted_class == real_label:
        print(f"{image_name} - Correcta: {real_label} (Predicción: {predicted_class})")
        correct_predictions += 1
    else:
        print(f"{image_name} - Incorrecta: {real_label} (Predicción: {predicted_class})")
        incorrect_predictions += 1

# Imprimir el resumen de las predicciones
print(f"Total de imágenes: {correct_predictions + incorrect_predictions}")
print(f"Correctamente clasificadas: {correct_predictions}")
print(f"Incorrectamente clasificadas: {incorrect_predictions}")
