# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io, filters, measure
from scipy import ndimage
from keras.models import Sequential
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import h5py
from generator import gather_images

# Find location of image files and labeled images
data = glob('data/raw/resized/with_people/splits/*hsv*.png')
labels = glob('data/processed/dots/with_people/splits/*.png')

# Split into the training and testing data
train_X, test_X = train_test_split(data, test_size=0.2, random_state=33)
train_Y, test_Y = train_test_split(labels, test_size=0.2, random_state=33)

# Select Batch Size
batch_size = 30

# Set up Convolutional Network
model = tf.keras.models.Sequential([
  tf.keras.layers.BatchNormalization(input_shape=(108,192,1)),
  tf.keras.layers.Conv2D(8, (5,5), padding="same", activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(3,3)),
  tf.keras.layers.Conv2D(16, (5,5), padding="same", activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(3,3)),
  tf.keras.layers.Conv2D(32, (2,2), padding="same", activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Conv2D(64, (2,2), padding="same", activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='linear'),
])

# Compile the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mean_squared_error', 'mean_absolute_error'])

# Set up early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Fit the model and save the history
h = model.fit_generator(gather_images(train_X, train_Y, batch_size=batch_size),
                    steps_per_epoch = len(train_X)//batch_size, epochs=50, 
                    validation_data=gather_images(test_X, test_Y, batch_size=batch_size), 
                    validation_steps = len(test_X)//batch_size, callbacks=[es])

# Save the model
model.save("model.h5")
print("Saved model to disk")

# Plot the result
error_mse = pd.DataFrame({'training_mse':h.history['mean_squared_error'], 
                       'testing_mse': h.history['val_mean_squared_error'], 
                       'epoch': range(1,len(h.history['mean_squared_error'])+1,1)})

error_mse.plot('epoch', ['training_mse', 'testing_mse'], figsize=(10,5))
plt.ylabel('Mean Squared Error (MSE)')
plt.show()