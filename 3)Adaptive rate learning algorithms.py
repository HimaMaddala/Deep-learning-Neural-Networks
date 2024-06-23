# importing modules 
import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Activation 
import matplotlib.pyplot as plt 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
  
# normalize image pixel values by dividing  
# by 255 
gray_scale = 255
x_train /= gray_scale 
x_test /= gray_scale 

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model = Sequential([ 
    Flatten(input_shape=(28, 28)), 
    
    Dense(256, activation='relu'),   

    Dense(128, activation='relu'),  

    Dense(10, activation='sigmoid'),   
]) 

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 

model.fit(x_train, y_train, epochs=50,callbacks=[callback],  
          batch_size=200,  
          validation_data = (x_val,y_val)) 

results = model.evaluate(x_test,  y_test, verbose = 0) 
print('test loss, test acc:', results)

model.compile(optimizer='SGD', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 

model.fit(x_train, y_train, epochs=50,callbacks=[callback],  
          batch_size=200,  
          validation_data = (x_val,y_val)) 

results = model.evaluate(x_test,  y_test, verbose = 0) 
print('test loss, test acc:', results)
