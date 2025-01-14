import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

# Parámetros de configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Ruta de las imágenes y archivo CSV
train_dir = "Train"
csv_file = "data_refinado.csv"

# Leer el archivo CSV
data = pd.read_csv(csv_file)


print("Dispositivos disponibles:")
print(tf.config.list_physical_devices('GPU'))

# Convertir etiquetas a enteros y agregar ruta completa de las imágenes
data['label'] = data['label'].astype(int)
data['image_path'] = data['image'].apply(lambda x: os.path.join(train_dir, x))

def preprocess_and_augment(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)  # Estandarizar imagen

    # # # # Aumentos
    img = tf.image.adjust_brightness(img, 0.1)
    img = tf.image.adjust_contrast(img, 5)
    img = tf.image.random_flip_up_down(img)
    
    return img, label



# Crear dataset de TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((data['image_path'].values, data['label'].values))
train_dataset = train_dataset.map(preprocess_and_augment).batch(BATCH_SIZE)

# Visualización de datos
def visualize_batch(dataset):
    for images, labels in dataset.take(1):  # Tomar un batch
        plt.figure(figsize=(10, 10))
        for i in range(9):  # Mostrar 9 imágenes
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"Etiqueta: {labels[i].numpy()}")
            plt.axis("off")
        plt.show()

# Llamar a la función para visualizar imágenes procesadas
visualize_batch(train_dataset)
