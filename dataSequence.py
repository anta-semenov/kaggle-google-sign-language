import os
import json
import random
import pandas as pd
from tensorflow.keras.utils import Sequence
import pyarrow.parquet as pq
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf



class ParquetSequence(Sequence):
    def __init__(self, root_folder, batch_size, indices, max_sequence_length = 32):
        self.root_folder = root_folder
        self.batch_size = batch_size
        self.data_df = pd.read_csv(os.path.join(root_folder, 'train.csv'))
        with open(os.path.join(root_folder, 'sign_to_prediction_index_map.json'), 'r') as f:
            self.sign_to_index = json.load(f)
        self.labels = self.data_df['sign']
        self.num_classes = len(self.sign_to_index)
        self.data_paths = self.data_df['path'].tolist()
        self.num_samples = len(self.data_paths)
        self.indices = indices
        self.ROWS_PER_FRAME = 543  # number of landmarks per frame
        self.max_sequence_length = max_sequence_length
    
    def __len__(self):
        return int(len(self.indices) / self.batch_size)
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(batch_indices)
    
    def on_epoch_end(self):
        random.shuffle(self.indices)

    def load_relevant_data_subset(self, pq_path):
        data_columns = ['x', 'y', 'z']
        data = pd.read_parquet(pq_path, columns=data_columns)
        n_frames = int(len(data) / self.ROWS_PER_FRAME)
        data = data.values.reshape(n_frames, self.ROWS_PER_FRAME, len(data_columns))
        return data.astype(np.float32)
    
    def __data_generation(self, batch_indices):
        X = []
        y = []
        max_frames = 0
        for i in batch_indices:
            data_path = os.path.join(self.root_folder, self.data_paths[i])
            data = self.load_relevant_data_subset(data_path)
            if len(data) < self.max_sequence_length:
                rows = self.max_sequence_length - len(data)
                data = np.append(data, np.array([np.zeros((self.ROWS_PER_FRAME, 3), dtype=np.float32)] * rows), axis=0)
            elif len(data) > self.max_sequence_length:
                start_index = (len(data) - self.max_sequence_length) // 2
                data = data[start_index: start_index + self.max_sequence_length]
            X.append(data)
            label = self.labels[i]
            y.append(self.sign_to_index[label])
            max_frames = max(max_frames, len(data))

            del data
            
        # pad all sequences to the same length            

        X = np.array(X)
        y = np.array(y)
        
        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)


def prepare_data(root_folder, batch_size, validation_split = 0.1, test_split = 0.1, max_sequence_length = 32):
        root_folder = root_folder
        batch_size = batch_size
        data_df = pd.read_csv(os.path.join(root_folder, 'train.csv'))
        with open(os.path.join(root_folder, 'sign_to_prediction_index_map.json'), 'r') as f:
            sign_to_index = json.load(f)
        labels = data_df['sign']
        num_classes = len(sign_to_index)
        data_paths = data_df['path'].tolist()
        num_samples = len(data_paths)
        # group labels by sign so for each sign we have a list of indices
        label_to_indices = {}
        for i, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)
        
        # split each sign's indices into train, validation, and test
        train_indices = []
        val_indices = []
        test_indices = []
        for label, indices in label_to_indices.items():
            random.shuffle(indices)
            num_train = int(len(indices) * (1 - validation_split - test_split))
            num_val = int(len(indices) * validation_split)
            train_indices.extend(indices[:num_train])
            val_indices.extend(indices[num_train:num_train+num_val])
            test_indices.extend(indices[num_train+num_val:])

        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        print(f"Number of training samples: {len(train_indices)}")
        print(f"Number of validation samples: {len(val_indices)}")

        return ParquetSequence(root_folder, batch_size, train_indices, max_sequence_length), ParquetSequence(root_folder, batch_size, val_indices, max_sequence_length), ParquetSequence(root_folder, batch_size, test_indices, max_sequence_length)

# write code to initialize the sequence and get the validation and test data examples
# and print their shapes and first element
# train_data, val_data, test_data = prepare_data('asl-signs', 256, 0.15, 0)

