# Dataset from https://www.kaggle.com/datasets/ashishguptajiit/handwritten-az?resource=download
#Note: this dataset is not included in the repo as it exceeds github's file size limit, but can be downloaded from the above link, simply add it to the folder after downloading the repo.
#Credit to https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/multi-class_classification_with_MNIST.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=multiclass_tf2-colab&hl=en#scrollTo=QF0BFRXTOeR3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

def create_model(training_rate: float):
    """Create a deep neural model with training rate."""
    model = tf.keras.models.Sequential()

    # Neural Network model
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=32, activation='relu'))
    
    # Dropout layer to reduce overfitting
    model.add(tf.keras.layers.Dropout(rate=0.3))
    
    #output layer, 26 units for each letter, one for each letter
    model.add(layers.Dense(units=26, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_rate),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])
    
    return model

def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    """Train the model by feeding it data."""

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                        epochs=epochs, shuffle=True, 
                        validation_split=validation_split)
    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch. 
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""  
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
  plt.show()
  
def shuffle_in_unison(arr1, arr2):
    """Shuffle two arrays in the same order."""
    rng_state = np.random.get_state()
    np.random.shuffle(arr1)
    np.random.set_state(rng_state)
    np.random.shuffle(arr2)

if __name__ == "__main__":
    #import data
    data = pd.read_csv('Handwriting/handwritten_data_785.csv', sep=',')
    
    #split data into features and labels, first index is the label, rest is the feature
    features = data.iloc[:, 1:].values
    labels = data.iloc[:, 0].values
    
    #shuffle data
    shuffle_in_unison(features, labels)
    
    #normalize data
    features = features / 255
    
    #split data into training and testing data
    x_train = features[:round(features.shape[0]*0.8)]
    y_train = labels[:round(features.shape[0]*0.8)]
    x_test = features[round(features.shape[0]*0.8):]
    y_test = labels[round(features.shape[0]*0.8):]
    
    # The following variables are the hyperparameters.
    learning_rate = 0.003
    epochs = 50
    batch_size = 4000
    validation_split = 0.2

    # Establish the model's topography.
    my_model = create_model(learning_rate)

    # Train the model on the normalized training set.
    epochs, hist = train_model(my_model, x_train, y_train, 
                            epochs, batch_size, validation_split)

    # Plot a graph of the metric vs. epochs.
    list_of_metrics_to_plot = ['accuracy']
    plot_curve(epochs, hist, list_of_metrics_to_plot)

    # Evaluate against the test set.
    print("\n Evaluate the new model against the test set:")
    my_model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

        