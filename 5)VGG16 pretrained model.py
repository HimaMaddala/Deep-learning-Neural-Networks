import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Load pre-trained VGGNet-16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False
# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
import matplotlib.pyplot as plt
import numpy as np
# Select an image from the test set
#17 horse
index = 12  # You can change this index to choose a different image
image = x_test[index]
true_label = y_test[index]

# Reshape the image to (1, 32, 32, 3) to match model input shape
image_for_prediction = np.expand_dims(image, axis=0)

# Make predictions using the model
predictions = model.predict(image_for_prediction)

# Get the predicted class label
predicted_class = np.argmax(predictions)

# Define class labels for CIFAR-10 dataset
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

# Display the image and predicted label
plt.imshow(image)
plt.title(f"True Label: {class_labels[np.argmax(true_label)]}, Predicted Label: {class_labels[predicted_class]}")
plt.axis('off')
plt.show()

predictions = Dense(10, activation='softmax')(x)

# Create the transfer learning model
model = Model(inputs=base_model.input, outputs=predictions)
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
