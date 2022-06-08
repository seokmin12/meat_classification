import tensorflow as tf

model = tf.keras.models.load_model('model/model4.h5')
model.summary()
