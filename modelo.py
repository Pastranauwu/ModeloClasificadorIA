import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Ruta de las imágenes y archivo CSV
train_dir = "Train"
csv_file = "data_refinado.csv"

# Leer datos y preparar paths
data = pd.read_csv(csv_file)
data['label'] = data['label'].astype(int)
data['image_path'] = data['image'].apply(lambda x: os.path.join(train_dir, x))

# Dividir en conjuntos de entrenamiento y validación
train_data, val_data = train_test_split(data, test_size=0.1, random_state=48)

IMG_SIZE = (380, 380)  # Tamaño recomendado para EfficientNetB4
BATCH_SIZE = 16  # Tamaño reducido para manejar el modelo más grande
EPOCHS = 50

# Calcular pesos de clases
class_weights = compute_class_weight('balanced', classes=np.unique(train_data['label']), y=train_data['label'])
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Preprocesamiento y aumentos optimizados
def preprocess_and_augment(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)

    # Aumentos simplificados
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label

def preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img, label

# Crear datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_data['image_path'].values, train_data['label'].values))
train_dataset = (train_dataset
                 .map(preprocess_and_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                 .shuffle(2000)
                 .batch(BATCH_SIZE)
                 .cache()
                 .prefetch(tf.data.experimental.AUTOTUNE))

val_dataset = tf.data.Dataset.from_tensor_slices((val_data['image_path'].values, val_data['label'].values))
val_dataset = (val_dataset
               .map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .batch(BATCH_SIZE)
               .cache()
               .prefetch(tf.data.experimental.AUTOTUNE))

# Modelo con EfficientNetB4
def create_model():
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model, base_model

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model_efficientnetb4.h5', monitor='val_auc', save_best_only=True, mode='max')
]

# Configurar distribución
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model, base_model = create_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])

# Entrenamiento inicial
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS,
                    callbacks=callbacks, class_weight=class_weights)

# Fine-tuning
def fine_tune_model(model, base_model):
    base_model.trainable = True
    for layer in base_model.layers[:50]:  # Mantener congeladas algunas capas
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

with strategy.scope():
    model = fine_tune_model(model, base_model)
    history_finetune = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS,
                                 callbacks=callbacks, class_weight=class_weights)

# Evaluación final
val_loss, val_accuracy, val_auc, val_precision, val_recall = model.evaluate(val_dataset)
print(f"Post-fine-tuning - Loss: {val_loss}, Accuracy: {val_accuracy}, AUC: {val_auc}, Precision: {val_precision}, Recall: {val_recall}")
