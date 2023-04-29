import tensorflow as tf
from tensorflow.keras.layers import Input, Activation

# Define the path to the Keras H5 model
keras_model_path = 'models/256_2_relu_lstm_1_512_swish_dominant_miror_v2/model_checkpoint-70.h5'

# Load the Keras model
keras_model = tf.keras.models.load_model(keras_model_path)

inputs = Input(shape=(543, 3), name='inputs')
x = keras_model(tf.expand_dims(inputs,0))
outputs = Activation('linear', name='outputs')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.summary(expand_nested=True)

# Convert the Keras model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the TFLite model to disk
tflite_model_path = '256_2_relu_lstm_1_512_swish_dominant_miror_norm.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)