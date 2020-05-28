from keras.datasets import mnist 
dataset=mnist.load_data('mymnist.db')
train , test = dataset
X_train , Y_train = train
X_test ,Y_test = test
import matplotlib.pyplot as plt
X_train_1d=X_train.reshape( -1 ,28*28)
X_test_1d=X_test.reshape(-1 ,28*28)
from keras.utils.np_utils import to_categorical
Y_train_cat=to_categorical(Y_train)
Y_test_cat=to_categorical(Y_test)
from keras.models import Sequential 
from keras.layers import Dense 
model = Sequential()
model.add(Dense(units=512 ,input_dim=28*28 ,activation='relu'))
model.add(Dense(units=256 , activation='relu'))
model.add(Dense(units=10 , activation = 'softmax'))
from keras.optimizers import RMSprop
model.compile(optimizer =RMSprop() , loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train ,Y_train_cat ,epochs=100)
