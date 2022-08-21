#Credit to https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/multi-class_classification_with_MNIST.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=multiclass_tf2-colab&hl=en#scrollTo=QF0BFRXTOeR3
#https://thecleverprogrammer.com/2022/01/03/stock-price-prediction-with-lstm/
from random import shuffle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_model(training_rate: float, len1: int):
    """Create a deep neural model with training rate."""
    model = tf.keras.models.Sequential()
    
    #LSTM model
    model.add(layers.LSTM(units=128, return_sequences=True, input_shape=(len1, 1)))
    model.add(layers.LSTM(units=64, return_sequences=False))
    model.add(layers.Dense(units=1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    
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


if __name__ == "__main__":
    stock_name = 'TSLA'
    stock_details = yf.Ticker(stock_name)
    stock_history = stock_details.history(period='7d', interval='1m')
    stock_history.reset_index(inplace=True)
    df = stock_history.copy()
    df = df.drop(columns=["High", "Low", "Dividends", "Stock Splits", "Datetime"])
    x = df[["Open", "Volume"]]
    y = df["Close"]
    x = x.to_numpy()
    y = y.to_numpy()
    y = y.reshape(-1, 1)
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)-0.001
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(x_train)
    
    # The following variables are the hyperparameters.
    learning_rate = 0.003
    epochs = 50
    batch_size = 4000
    validation_split = 0.2

    # Establish the model's topography.
    my_model = create_model(learning_rate, x_train.shape[1])

    # Train the model on the normalized training set.
    epochs, hist = train_model(my_model, x_train, y_train, 
                            epochs, batch_size, validation_split)

    # Evaluate against the test set.
    print("\n Evaluate the new model against the test set:")
    my_model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
    
    last_feature = np.array([x_test[-1]])
    predicted_price = my_model.predict(last_feature)
    predicted_price = scaler.inverse_transform(predicted_price)
    print("\n The predicted price is:", predicted_price)
        