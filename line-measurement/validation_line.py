#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
#import keras
import numpy as np
#model = keras.models.load_model('line_measure_keras')
#testdata=np.array([4,267,146.86])
#print(testdata.shape)
#predictions = model.predict(testdata.reshape(1,3))
a = -0.07280067919587918
b = 0.0001767467766172589
c = -0.038716739957248475
d = 1.1437976876676706e-05
e = -0.020572336452301287
f = 11.628985207409308
xm1=5
xm2=135.267623503373
xm3=198.305320150519
prediction2=(a*xm1**2+b*xm1)+(b*xm2**2+c*xm2)+(d*xm3**2+e*xm3)+f

#print(predictions)
print(prediction2)