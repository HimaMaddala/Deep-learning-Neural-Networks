import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

# Generate a dataset
x = np.linspace(-10, 10, 100)
y = x**3 + np.random.normal(0, 10, 100)

plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Dataset')
plt.show()

# Scale the input data
scaler = StandardScaler()
x = scaler.fit_transform(x.reshape(-1, 1))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create a neural network without regularization
model_no_reg = Sequential([
    Dense(100, input_shape=(1,), activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1)
])

# Compile the model
model_no_reg.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model_no_reg.fit(x_train, y_train, epochs=500, batch_size=32, verbose=0)

# Evaluate the model
training_loss = model_no_reg.evaluate(x_train, y_train, verbose=0)
test_loss = model_no_reg.evaluate(x_test, y_test, verbose=0)

print(f'Neural Network without Regularization:')
print(f'Training loss: {training_loss:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Plot the true vs. predicted values for the neural network without regularization
y_pred = model_no_reg.predict(x_test)
plt.scatter(x_test, y_test, label='True')
plt.scatter(x_test, y_pred, label='Predicted')
plt.legend()
plt.title('Neural Network without Regularization')
plt.show()

# Create a neural network with L1 regularization
model_L1 = Sequential([
    Dense(100, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(1)
])

# Compile the model
model_L1.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model_L1.fit(x_train, y_train, epochs=500, batch_size=32, verbose=0)

# Evaluate the model
training_loss = model_L1.evaluate(x_train, y_train, verbose=0)
test_loss = model_L1.evaluate(x_test, y_test, verbose=0)

print(f'Neural Network with L1 Regularization:')
print(f'Training loss: {training_loss:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Plot the true vs. predicted values for the neural network with L1 regularization
y_pred = model_L1.predict(x_test)
plt.scatter(x_test, y_test, label='True')
plt.scatter(x_test, y_pred, label='Predicted')
plt.legend()
plt.title('Neural Network with L1 Regularization')
plt.show()


# Evaluate the model without regularization
training_loss_no_reg = model_no_reg.evaluate(x_train, y_train, verbose=0)
test_loss_no_reg = model_no_reg.evaluate(x_test, y_test, verbose=0)

# Evaluate the model with L1 regularization
training_loss_L1 = model_L1.evaluate(x_train, y_train, verbose=0)
test_loss_L1 = model_L1.evaluate(x_test, y_test, verbose=0)

# Print losses before and after regularization
print('Before Regularization:')
print(f'Neural Network without Regularization:')
print(f'Training loss: {training_loss_no_reg:.4f}')
print(f'Test loss: {test_loss_no_reg:.4f}')

print('After L1 Regularization:')
print(f'Neural Network with L1 Regularization:')
print(f'Training loss: {training_loss_L1:.4f}')
print(f'Test loss: {test_loss_L1:.4f}')

# Create a neural network with L2 regularization
model_L2 = Sequential([
    Dense(100, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(1)
])

# Compile the model
model_L2.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model_L2.fit(x_train, y_train, epochs=500, batch_size=32, verbose=0)

# Evaluate the model
training_loss = model_L2.evaluate(x_train, y_train, verbose=0)
test_loss = model_L2.evaluate(x_test, y_test, verbose=0)

print(f'Neural Network with L2 Regularization:')
print(f'Training loss: {training_loss:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Plot the true vs. predicted values for the neural network with L2 regularization
y_pred = model_L2.predict(x_test)
plt.scatter(x_test, y_test, label='True')
plt.scatter(x_test, y_pred, label='Predicted')
plt.legend()
plt.title('Neural Network with L2 Regularization')
plt.show()

# Ridge Regression
param_grid = {'alpha': np.arange(0.1, 0.5, 0.01)}
rand_grid_search_ridge = GridSearchCV(Ridge(), param_grid, cv=5)
rand_grid_search_ridge.fit(x_train.reshape(-1, 1), y_train)
print(f'Ridge Regression:')
print(f'Best alpha: {rand_grid_search_ridge.best_params_["alpha"]:.2f}')
print(f'Test score: {rand_grid_search_ridge.score(x_test.reshape(-1, 1), y_test):.2f}')

# Lasso Regression
rand_grid_search_lasso = RandomizedSearchCV(Lasso(), param_grid, cv=5)
rand_grid_search_lasso.fit(x_train.reshape(-1, 1), y_train)
print(f'Lasso Regression:')
print(f'Best alpha: {rand_grid_search_lasso.best_params_["alpha"]:.2f}')
print(f'Test score: {rand_grid_search_lasso.score(x_test.reshape(-1, 1), y_test):.2f}')
