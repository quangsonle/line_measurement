
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
...
# load dataset
dataframe = pd.read_csv("frame_test.csv")
#dataset = dataframe.values
#print(dataset)
print(dataframe.head)
# split into input (X) and output (Y) variables
#X = dataset[:,0:2]
#print(X.shape)
#Y = dataset[:,3]
Y=pd.DataFrame(dataframe,columns=['dis'])
X=dataframe.drop('dis',axis=1)
model = Sequential()
model.add(Dense(20, input_dim=3, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))
# Compile model
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# Fit the model
history = model.fit(X, Y, validation_split=0.2, epochs=100)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('line_measure_keras')