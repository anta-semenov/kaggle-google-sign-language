# write a function for reading parquet file into pandas dataframe
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LayerNormalization, MultiHeadAttention, Concatenate, GlobalAveragePooling1D, Masking, LSTM, SimpleRNN, GRU, Activation, LSTMCell, SimpleRNNCell, StackedRNNCells, RNN, GRUCell
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, schedules
from dataSequence import prepare_data 
import numpy as np
import datetime
import keras_tuner as kt

def read_parquet(path):
    df = pd.read_parquet(path)
    return df




# Define the input layer
# input_layer = Input(shape=(None, 543, 3))

# def replace_nan_with_zero(x):
#     return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

# temp = tf.nest.map_structure(replace_nan_with_zero, input_layer)
# indices = [0, 61, 17, 291] + list(range(468, 488)) + [489] + list(range(522, 542))
# temp = tf.gather(temp, indices, axis=2)
# temp = temp[:, :, :, :2]

# tensor_shape = tf.shape(temp)
# # flatten the last dimension
# temp = tf.reshape(temp, [tensor_shape[0], tensor_shape[1], 90])
# x = Masking(mask_value=0.0)(temp)

# def dence_block(units, activation = 'swish'):
#     dence = Dense(units)
#     norm = LayerNormalization()
#     activation = Activation('swish')
#     dropout = Dropout(0.2)
#     return lambda x: dropout(activation(norm(dence(x))))

# dense_units = [128, 128]

# for units in dense_units:
#     x = dence_block(units)(x)

# rnn_stack_layer = RNN(StackedRNNCells([LSTMCell(128) for _ in range(1)]))

# # x = LSTM(128)(x)
# # x = rnn_stack_layer(x)

# x = dence_block(128)(x)

# outputs = Dense(units=250, activation='softmax')(x)

def model_builder(dense_units, dense_block_layers, dense_block_activation, rnn_type, rnn_layers, out_dense_units, out_dense_activation):
    input_layer = Input(shape=(None, 543, 3), name='inputs')

    left_hand_indixes = list(range(468, 488))
    right_hand_indixes = list(range(522, 542))
    face_and_pose_indixes = [0, 61, 17, 291, 489]
    landmarks_count = len(face_and_pose_indixes + left_hand_indixes + right_hand_indixes)
    left_hand_sum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(input_layer, left_hand_indixes, axis=2))))
    right_hand_sum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(input_layer, right_hand_indixes, axis=2))))
    face_and_pose = tf.gather(input_layer, face_and_pose_indixes, axis=2)[:, :, :, :2]
    left_hand = tf.gather(input_layer, left_hand_indixes, axis=2)[:, :, :, :2]
    rigth_hand = tf.gather(input_layer, right_hand_indixes, axis=2)[:, :, :, :2]

    first_hand_x = tf.where(left_hand_sum > right_hand_sum, tf.subtract(1.0, left_hand[:, :, :, :1]), rigth_hand[:, :, :, :1])
    first_hand_y = tf.where(left_hand_sum > right_hand_sum, left_hand[:, :, :, 1:], rigth_hand[:, :, :, 1:])
    first_hand = tf.squeeze(tf.stack([first_hand_x, first_hand_y], axis=-1), axis=[-2])
    first_hand_normalized = tf.linalg.normalize(first_hand, axis=-2)[0]

    
    second_hand_x = tf.where(left_hand_sum > right_hand_sum, tf.subtract(1.0, rigth_hand[:, :, :, :1]), left_hand[:, :, :, :1])
    second_hand_y = tf.where(left_hand_sum > right_hand_sum, rigth_hand[:, :, :, 1:], left_hand[:, :, :, 1:])
    second_hand = tf.squeeze(tf.stack([second_hand_x, second_hand_y], axis=-1), axis=[-2])  
    second_hand_normalized = tf.linalg.normalize(second_hand, axis=-2)[0]
    
    x = tf.concat([face_and_pose, first_hand, second_hand, first_hand_normalized, second_hand_normalized], axis=-2)

    tensor_shape = tf.shape(x)

    # x_coord = tf.where(left_hand_sum > right_hand_sum, tf.subtract(1.0, x[:, :, :, :1]), x[:, :, :, :1])
    # # x_coord = tf.linalg.normalize(x_coord, axis=-1)[0]
    # y_coord = x[:, :, :, 1:]
    # # y_coord = tf.linalg.normalize(y_coord, axis=-1)[0]

    # x = tf.squeeze(tf.stack([x_coord, y_coord], axis=-1), axis=[-2])

    def replace_nan_with_zero(x):
        return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    x = tf.nest.map_structure(replace_nan_with_zero, x)

    # flatten the last dimension
    x = tf.reshape(x, [tensor_shape[0], tensor_shape[1], 170])
    x = Masking(mask_value=0.0)(x)

    def dence_block(units, activation = 'swish'):
        dence = Dense(units)
        norm = LayerNormalization()
        activation = Activation('swish')
        dropout = Dropout(0.2)
        return lambda i: dropout(activation(norm(dence(i))))


    for _ in range(dense_block_layers):
        x = dence_block(dense_units, dense_block_activation)(x)
    # rnn_units = hp.Choice('rnn_units', values=units_variations)
    if rnn_layers == 1:
        if rnn_type == 'rnn':
            x = SimpleRNN(dense_units)(x)
        elif rnn_type == 'lstm':
            x = LSTM(dense_units)(x)
        elif rnn_type == 'gru':
            x = GRU(dense_units)(x)
    else :
        if rnn_type == 'rnn':
            cell = SimpleRNNCell(dense_units)
        elif rnn_type == 'lstm':
            cell = LSTMCell(dense_units)
        elif rnn_type == 'gru':
            cell = GRUCell(dense_units)
        rnn_stack_layer = RNN(StackedRNNCells([cell for _ in range(rnn_layers)]))
        x = rnn_stack_layer(x)

    x = dence_block(out_dense_units, out_dense_activation)(x)

    outputs = Dense(units=250, activation='softmax', name='outputs')(x)

    model = Model(inputs=input_layer, outputs=outputs)

    boundaries = [331 * n for n in [30, 45, 55, 65, 75]]
    print(boundaries)
    values = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
    lr_sched = schedules.PiecewiseConstantDecay(boundaries, values)

    optimizer = Adam(lr_sched)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


# # Add an embedding layer
# embedding_layer = Embedding(input_dim=40, output_dim=128)(input_layer)

# # Add a positional encoding layer
# seq_length = tf.shape(embedding_layer)[1]
# position = tf.range(0, seq_length)
# position_enc = Embedding(input_dim=seq_length, output_dim=128)(position)
# position_enc = tf.expand_dims(position_enc, axis=0)
# position_enc = tf.tile(position_enc, multiples=[tf.shape(embedding_layer)[0], 1, 1])
# embedding_layer = embedding_layer + position_enc

# # Add a multi-head attention layer
# attention_layer = MultiHeadAttention(num_heads=8, key_dim=128)(embedding_layer, embedding_layer)
# attention_layer = LayerNormalization(epsilon=1e-6)(attention_layer + embedding_layer)

# # Add a feedforward layer
# feedforward_layer = Dense(units=512, activation='relu')(attention_layer)
# feedforward_layer = Dropout(rate=0.1)(feedforward_layer)
# feedforward_layer = Dense(units=128)(feedforward_layer)
# feedforward_layer = Dropout(rate=0.1)(feedforward_layer)

# # Add a global average pooling layer
# pooling_layer = GlobalAveragePooling1D()(feedforward_layer)

# # Add an output layer
# output_layer = Dense(units=250, activation='softmax')(pooling_layer)

# Create the model
# model = Model(inputs=input_layer, outputs=outputs)

# boundaries = [train_data.__len__() * n for n in [30, 45, 55, 65, 75]]
# print(boundaries)
# values = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
# lr_sched = schedules.PiecewiseConstantDecay(boundaries, values)

# optimizer = Adam(lr_sched)

# # Compile the model
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # Print the model summary
# model = tf.keras.models.load_model('models/lstm128/model_checkpoint-10.h5')
# model.summary()

# log_dir = "logs/fit_lstm/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# tuner = kt.BayesianOptimization(model_builder,
#                      objective='val_accuracy',
#                      max_trials=50,
#                      directory='model_tune_bayes',
#                      project_name='asl_model_search')
save_model = tf.keras.callbacks.ModelCheckpoint(filepath='models/256_2_relu_lstm_1_512_swish_dominant_miror_v2/model_checkpoint-{epoch:02d}.h5')
# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Train the model
dense_units = 256
dense_block_layers = 2
dense_block_activation = 'relu'
rnn_type = 'lstm'
rnn_layers = 1
out_dense_units = 512
out_dense_activation = 'swish'
out_dense_block_layers = 1
model = model_builder(dense_units, dense_block_layers, dense_block_activation, rnn_type, rnn_layers, out_dense_units, out_dense_activation)
train_data, val_data, test_data = prepare_data('asl-signs', 256, 0.1, 0, 32)
model.fit(train_data, validation_data=val_data, epochs=70, callbacks=[save_model])
