import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Cast the records into float values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize image pixel values by dividing by 255
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

# Splitting the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# Define the model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.2),  # Adding dropout with a dropout rate of 0.2
    Dense(128, activation='relu'),
    Dropout(0.2),  # Adding dropout with a dropout rate of 0.2
    Dense(10, activation='softmax'),  # Using softmax for multiclass classification
])

# Specify the optimizer
optimizer = Adam()

#from tensorflow.keras.optimizers import RMSprop, Adagrad, SGD, Adam, Adadelta
# Example 1: RMSprop optimizer with default parameters
# optimizer = RMSprop()

# Example 2: Adagrad optimizer with default parameters
# optimizer = Adagrad()

# Example 3: SGD optimizer with default parameters
# optimizer = SGD()

# Example 4: Adam optimizer with default parameters
#optimizer = Adam()

# Example 5: Adadelta optimizer with default parameters
# optimizer = Adadelta()

# Compile the model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',  # Using sparse categorical crossentropy for integer labels
              metrics=['accuracy'])

# Setup early stopping callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model with early stopping callback
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    callbacks=[callback])

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test,verbose = 0)
print("Test Accuracy:", test_acc)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_acc = model.evaluate(x_test, y_test,verbose = 0)
print("Test Loss,Test Accuracy", test_acc)

fig, ax = plt.subplots(10, 10) 
k = 0
for i in range(10): 
    for j in range(10): 
        ax[i][j].imshow(x_train[k].reshape(28, 28),  
                        aspect='auto') 
        k += 1
plt.show() 
