import keras.callbacks
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path for exported data, numpy array
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# videos are going to be 30 frames in length
sequence_length = 30

# create labels for each category of actions
label_map = {label: num for num, label in enumerate(actions)}

# load in data and apply label to each set of sequences
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
Y = to_categorical(labels).astype(int)

# split into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

# Build and Train LSTM Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# logging directory
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


class stopTrainingCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('categorical_accuracy') > 0.95:
            print("\nReached %2.2f%% accuracy, so stopping training!!" % 95)
            self.model.stop_training = True


callbacks = stopTrainingCallBack()

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, Y_train, epochs=500, callbacks=[callbacks])
model.save('action.h5')

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_train)

ytrue = np.argmax(Y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
matrix_results = multilabel_confusion_matrix(ytrue, yhat)
print(matrix_results)
acc_results = accuracy_score(ytrue, yhat)
print(acc_results)
