import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# === PARÁMETROS ===
DATASET_DIR = './dataset'  # <-- Cambia esto
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
FINE_TUNE_AT = 100  # Número de capas desde donde empezamos a descongelar

# === 1. Cargar generadores (misma división 80/20) ===
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    shuffle=False
)

# === 2. Reconstruir modelo (cargar pesos del modelo anterior entrenado) ===
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = True

# 3. Congelar todas excepto las últimas FINE_TUNE_AT capas
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# 4. Añadir head igual que antes
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# 5. Compilar con tasa de aprendizaje baja
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Reentrenar (fine-tuning)
model.fit(train_gen, epochs=5, validation_data=val_gen)

# 7. Evaluación final
loss, accuracy = model.evaluate(val_gen)
print(f"\n🔍 Nueva precisión después de fine-tuning: {accuracy:.4f}")

# 8. Matriz de confusión y reporte
y_true = val_gen.classes
y_pred_prob = model.predict(val_gen)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\n📊 Reporte de Clasificación tras fine-tuning:")
print(classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys()))

print("\n🧮 Matriz de Confusión:")
print(confusion_matrix(y_true, y_pred))

# (Opcional) Guardar el nuevo modelo fine-tuned
model.save("first_model.keras")

