import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

x_vals = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_vals = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(x_vals, y_vals, epochs=1000)

print(model.predict([19.0]))