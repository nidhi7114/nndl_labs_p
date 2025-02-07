import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load dataset
csv_path = "C:\\Users\\Administrator\\NNDL_LABS\\LAB3\\pokemon.csv"
image_dir = "C:\\Users\\Administrator\\NNDL_LABS\\LAB3images"
df = pd.read_csv(csv_path)
df = df.dropna(subset=["Type1"])
df["Type1"] = df["Type1"].astype(str)

# Parameters
img_size = (128, 128)
batch_size = 32
num_classes = df["Type1"].nunique()

# Data generators
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col="Name",
    y_col="Type1",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)
val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col="Name",
    y_col="Type1",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(img_size[0], img_size[1], 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate on test set
test_loss, test_acc = model.evaluate(val_generator)
print(f'Test Accuracy: {test_acc:.2f}')
