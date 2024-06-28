import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense 
 
# Load the IMDb dataset 
max_features = 5000  # Only consider the top 5000 words in the dataset 
max_len = 300  # Cut reviews after 300 words 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) 
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, 
maxlen=max_len) 
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len) 
 
# Build the LSTM model 
model = Sequential() 
model.add(Embedding(input_dim=max_features, output_dim=128, 
input_length=max_len)) 
model.add(LSTM(64, return_sequences=True)) 
model.add(LSTM(64)) 
model.add(Dense(1, activation='sigmoid')) 
 
# Compile the model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
 
# Train the model 
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2) 
 
# Evaluate the model on the test set 
test_loss, test_accuracy = model.evaluate(x_test, y_test) 
print(f'Test Accuracy: {test_accuracy * 100:.2f}%') 

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

t = Tokenizer()
sample_review = "This movie was fantastic! I loved every moment of it."

# Convert the sample review into a list containing one element (the review)
sample_sequence = t.texts_to_sequences([sample_review])

# Since the model expects sequences of fixed length, you might need to pad the sequence
sample_sequence = pad_sequences(sample_sequence, maxlen=10)

# Predict sentiment
y_pred = model.predict(sample_sequence)

# Output the prediction
if y_pred > 0.5:
    print("Positive review!")
else:
    print("Negative review.")
