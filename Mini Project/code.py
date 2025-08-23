# Step 1: Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# Step 2: Sample Dataset (You can replace this with a CSV file with 'text' and 'label' columns)
texts = [
 "I love this product", "This is the best!", "Absolutely wonderful experience", "Worst service ever", "I hate this item", "Very bad quality",  "Not bad at all", "Quite good", "Could be better", "Totally unacceptable"
]
labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0] # 1 = Positive, 0 = Negative
# Step 3: Preprocessing
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post', maxlen=10)
2117230020142
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)
# Step 4: Build LSTM Model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64, input_length=10))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Step 5: Train the Model
model.fit(np.array(X_train), np.array(y_train), epochs=5, validation_data=(np.array(X_test),
np.array(y_test)))
# Step 6: Test Prediction
test_text = ["I am very happy with this", "Terrible experience"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=10, padding='post')
predictions = model.predict(test_pad)
# Output predictions
for i, sentence in enumerate(test_text):
 sentiment = "Positive" if predictions[i] > 0.5 else "Negative"
 print(f"Text: \"{sentence}\" => Sentiment: {sentiment} ({predictions[i][0]:.2f})")