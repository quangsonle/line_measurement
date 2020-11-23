from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras
import numpy as np
model = keras.models.load_model('line_measure_keras')
testdata=np.array([4,146.8,267])
print(testdata.shape)
predictions = model.predict(testdata.reshape(1,3))
print(predictions)
