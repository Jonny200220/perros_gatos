import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# 1. Parámetros
DATASET_DIR = './dataset'  # <- cambia esto a la ruta real
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

# 2. Crear generadores de datos (80/20 aleatorio)
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    shuffle=False  # Para evaluar luego
)

# 3. Cargar MobileNetV2 sin la parte superior
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # 4. Congelar todas las capas del modelo base

# 5. Agregar nuevas capas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)  # 2 clases: salida binaria

model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compilar y entrenar solo el head
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5, validation_data=val_generator)

# 7. Evaluar con conjunto de prueba
loss, accuracy = model.evaluate(val_generator)
print(f"\nPrecisión en validación: {accuracy:.4f}")

# 8. Matriz de confusión y reporte
y_true = val_generator.classes
y_pred_prob = model.predict(val_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_true, y_pred))