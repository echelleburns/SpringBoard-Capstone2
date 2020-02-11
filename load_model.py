# Import libraries
import tensorflow as tf

# Read model
model = tf.keras.models.load_model('model.h5')

# Get model summary
model.summary()