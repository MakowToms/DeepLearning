"""
Source: https://github.com/douglas125/SpeechCmdRecognition

A generator for reading and serving audio files

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

Remember to use multiprocessing:
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

"""

import numpy as np
import pandas as pd
import tensorflow.keras


class SpeechGen(tensorflow.keras.utils.Sequence):
    """
    'Generates data for Keras'

    list_IDs - list of files that this generator should load
    labels - dictionary of corresponding (integer) category
    to each file in list_IDs

    Expects list_IDs and labels to be of the same length
    """
    def __init__(self, list_IDs, labels, dataset_type, batch_size=32,
                 dim=16000, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.dataset_type = dataset_type
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        
        labels_df = pd.DataFrame(self.labels.items())
        counts = labels_df.groupby(1).count()[0]
        n_unknown = counts[0]
        self.n_in_one_class = None
        if counts.index.shape[0] == 1:
            # it's real test dataset - don't have to do anything with preparation
            pass
        else:
            unknown_labels = list(labels_df[labels_df.iloc[:, 1] == 0].iloc[:, 0])
            silence_labels = list(labels_df[labels_df.iloc[:, 1] == 1].iloc[:, 0])
            n_silence = counts[1]
            n_rest = counts.sum() - n_unknown - n_silence
            if n_rest == 0:
                # it's only silence task
                # balance silence and unknown
                self.list_IDs += silence_labels * int(len(unknown_labels)/6)
            else:
                # it's some of other datasets
                self.n_in_one_class = int(n_rest / counts.shape)
                # balance silence and other
                self.list_IDs += silence_labels * int(self.n_in_one_class/6)
                IDs_not_unknown = [ID for ID in self.list_IDs if not unknown_labels.__contains__(ID)]
                IDs_unknown = [ID for ID in self.list_IDs if unknown_labels.__contains__(ID)]
                assert len(self.list_IDs) == len(IDs_not_unknown) + len(IDs_unknown)
                self.list_IDs = IDs_not_unknown + IDs_unknown
                self.n = len(IDs_not_unknown) + self.n_in_one_class
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.n_in_one_class is None:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(np.floor(self.n / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.n_in_one_class is None: 
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle:
                np.random.shuffle(self.indexes)
        else:
            n_not_unknown = self.n - self.n_in_one_class
            self.indexes = np.zeros([self.n], dtype=np.intp)
            self.indexes[:n_not_unknown] = np.arange(n_not_unknown)
            self.indexes[n_not_unknown:] = np.random.choice(np.arange(n_not_unknown, len(self.list_IDs)), self.n_in_one_class, replace=False)
            if self.shuffle:
                np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim))
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # load data from file, saved as numpy array on disk
            curX = np.load(ID)

            # normalize
            # invMax = 1/(np.max(np.abs(curX))+1e-3)
            # curX *= invMax

            # curX could be bigger or smaller than self.dim
            if curX.shape[0] == self.dim:
                X[i] = curX
            elif curX.shape[0] > self.dim:  # bigger
                # we can choose any position in curX-self.dim
                if self.labels[ID] == 1:
                    if self.dataset_type == 'train':
                        randPos = np.random.randint(int((curX.shape[0]-self.dim)*3/5))
                    if self.dataset_type == 'val':
                        randPos = np.random.randint(int((curX.shape[0]-self.dim)*3/5), int((curX.shape[0]-self.dim)*4/5))
                    if self.dataset_type == 'test':
                        randPos = np.random.randint(int((curX.shape[0]-self.dim)*4/5), curX.shape[0]-self.dim)
                else:
                    randPos = np.random.randint(curX.shape[0]-self.dim)
                X[i] = curX[randPos:randPos+self.dim]
            else:  # smaller
                randPos = np.random.randint(self.dim-curX.shape[0])
                X[i, randPos:randPos + curX.shape[0]] = curX
                # print('File dim smaller')

            # Store class
            y[i] = self.labels[ID]

        return X, y
