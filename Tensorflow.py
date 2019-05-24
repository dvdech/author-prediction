#  Author: Dylan DeChiara & Stefanos Kalamaras

# Citations:
# https://stackoverflow.com/questions/11023411/how-to-import-csv-data-file-into-scikit-learn
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from __future__ import absolute_import, division, print_function

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

f1 = pd.read_csv("data.csv", header=0)

# organize data

id = f1.values[:,0]
sentences = f1.values[:,1]
words = list()

for sentence in sentences:
    for word in sentence.split():
        words.append(word)

authors = f1.values[:,2]
classes = ['EAP', 'HPL', 'MWS']



""" TENSORFLOW: PROJECT PT 2 """


for i in classes:
    print("There are", (authors == i).sum(), "examples of class", i)

# split data
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
train, test = next(splitter.split(sentences, authors))

# Verify class balance
for i in classes:
    print(i, (authors[train]==i).sum(), (authors[test]==i).sum())

# assign ints to authors
for i in range(len(authors)):
    author = authors[i]
    if author == 'EAP':
        authors[i] = 0
    elif author == 'HPL':
        authors[i] = 1
    else:
        authors[i] = 2

# transform data
count_vect = CountVectorizer()
count_sentences = count_vect.fit_transform(sentences)

print(authors[test])

# This is a perceptron
network = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation=tf.nn.softmax, input_shape=(25068, ))
])

# This is a shallow network
network_shallow = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(25068,)),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

# This is a deep network
network_deep = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(25068,)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

# This is a deeper network
network_deeper = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(25068,)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

""" shallow network config and train """

# Configure the network
network_shallow.compile(
    optimizer='adam',  # Algorithm for training the network
    metrics=['accuracy'],  # The way we want to measure performance
    loss='sparse_categorical_crossentropy'  # An internal metric that matches this dataset
)

# Train the network
val_acc_and_train_training_shallow = network_shallow.fit(
    count_sentences[train], authors[train],  # The training data
    epochs=32,  # How many times to go through the training data
    batch_size=32,  # Go through in batches for more efficient updates
    validation_data=(count_sentences[test], authors[test])  # The testing data
)

# Train the network
val_acc_training_shallow = network_shallow.fit(
    count_sentences[train], authors[train],  # The training data
    epochs=4,  # How many times to go through the training data
    batch_size=32,  # Go through in batches for more efficient updates
    validation_data=(count_sentences[test], authors[test])  # The testing data
)

""" shallow plot """

# Plot a learning curve and Training Curve w/ 32 epoch (shallow)
plt.plot(val_acc_and_train_training_shallow.epoch, val_acc_and_train_training_shallow.history['val_acc'], label="val_acc")
plt.plot(val_acc_and_train_training_shallow.epoch, val_acc_and_train_training_shallow.history['acc'], label="train_acc")

plt.title('Val_Acc and Train_Acc Curve Comparison With 32 Epoch (Shallow)')
plt.xlabel('Training epochs')
plt.ylabel('Network accuracy')
plt.legend()
plt.show()
plt.clf()

""" deep network config and train """

# Configure the network
network_deep.compile(
    optimizer='adam',  # Algorithm for training the network
    metrics=['accuracy'],  # The way we want to measure performance
    loss='sparse_categorical_crossentropy'  # An internal metric that matches this dataset
)

# Train the network
val_acc_and_train_training_deep = network_deep.fit(
    count_sentences[train], authors[train],  # The training data
    epochs=32,  # How many times to go through the training data
    batch_size=32,  # Go through in batches for more efficient updates
    validation_data=(count_sentences[test], authors[test])  # The testing data
)


""" deep network plot """

# Plot a learning curve and Training Curve w/ 32 epoch (deep)
plt.plot(val_acc_and_train_training_deep.epoch, val_acc_and_train_training_deep.history['val_acc'], label="val_acc")
plt.plot(val_acc_and_train_training_deep.epoch, val_acc_and_train_training_deep.history['acc'], label="train_acc")

plt.title('Val_Acc and Train_Acc Curve Comparison With 32 Epoch (Deep)')
plt.xlabel('Training epochs')
plt.ylabel('Network accuracy')
plt.legend()
plt.show()
plt.clf()


""" deeper network config and train """

# Configure the network
network_deeper.compile(
    optimizer='adam',  # Algorithm for training the network
    metrics=['accuracy'],  # The way we want to measure performance
    loss='sparse_categorical_crossentropy'  # An internal metric that matches this dataset
)

# Train the network
val_acc_and_train_training_deeper = network_deeper.fit(
    count_sentences[train], authors[train],  # The training data
    epochs=32,  # How many times to go through the training data
    batch_size=32,  # Go through in batches for more efficient updates
    validation_data=(count_sentences[test], authors[test])  # The testing data
)


""" Deeper network plots """

# Plot a learning curve and Training Curve w/ 32 epoch (deeper)
plt.plot(val_acc_and_train_training_deeper.epoch, val_acc_and_train_training_deeper.history['val_acc'], label="val_acc")
plt.plot(val_acc_and_train_training_deeper.epoch, val_acc_and_train_training_deeper.history['acc'], label="train_acc")

plt.title('Val_Acc and Train_Acc Curve Comparison With 32 Epoch (Deeper)')
plt.xlabel('Training epochs')
plt.ylabel('Network accuracy')
plt.legend()
plt.show()
plt.clf()


""" All 3 Val_Acc Compared """

plt.plot(val_acc_and_train_training_deeper.epoch, val_acc_and_train_training_deeper.history['val_acc'], label="val_acc_deeper")
plt.plot(val_acc_and_train_training_deep.epoch, val_acc_and_train_training_deep.history['val_acc'], label="val_acc_deep")
plt.plot(val_acc_and_train_training_shallow.epoch, val_acc_and_train_training_shallow.history['val_acc'], label="val_acc_shallow")

plt.title('All 3 Val_Acc curves compared')
plt.xlabel('Training epochs')
plt.ylabel('Network accuracy')
plt.legend()
plt.show()
plt.clf()