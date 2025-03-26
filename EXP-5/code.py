import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
plt.title(f"Label: {y_train[0]}")
plt.imshow(x_train[0,:,:]);

def preprocess(trainData, trainLabel, depth=10):
    x = tf.cast(trainData, dtype=tf.float32) / 255.0
    y = tf.cast(tf.one_hot(trainLabel, depth=depth), dtype=tf.int64)
    return x, y

Xtrain = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
XVal = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

model3x3 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='linear')
])

model5x5 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='linear')
])

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        self.epoch_times.append(elapsed_time)

# Training Setup
model3x3.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                 metrics=['accuracy'])

model5x5.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                 metrics=['accuracy'])

# Model Training
time3x3 = TimeHistory()
callback3x3 = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, verbose=1), time3x3]

history3x3 = model3x3.fit(Xtrain.repeat(),
                          validation_data=XVal.repeat(),
                          epochs=50,
                          steps_per_epoch=469,
                          validation_steps=50,
                          callbacks=callback3x3)

time5x5 = TimeHistory()
callback5x5 = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, verbose=1), time5x5]

history5x5 = model5x5.fit(Xtrain.repeat(),
                          validation_data=XVal.repeat(),
                          epochs=50,
                          steps_per_epoch=469,
                          validation_steps=25,
                          callbacks=callback5x5)

# Plotting Results
epochs = range(1, len(time3x3.epoch_times) + 1)

plt.figure(figsize=(15, 8))
plt.plot(epochs, history3x3.history["accuracy"], label="3x3 Filter with Adam", marker='o', color='r')
plt.plot(epochs, history5x5.history["accuracy"], label="5x5 Filter with RMSprop", marker='s', color='#FFA500')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.legend()
plt.show()
