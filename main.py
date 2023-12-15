# Importing necessary libraries
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Load the dataset
movie_reviews_data = pd.read_csv("IMDB Dataset.csv")

# Exploratory Data Analysis (EDA)
print("Shape of the dataset:")
print(movie_reviews_data.shape)
print("Null values in the dataset:\n")
print(movie_reviews_data.isnull().sum())
print("Summary statistics:\n")
print(movie_reviews_data.describe())
print("Info about the dataset:\n")
print(movie_reviews_data.info())
print("Unique values in 'sentiment' column:")
print(movie_reviews_data['sentiment'].unique())
print("Value counts in 'sentiment' column:\n")
print(movie_reviews_data['sentiment'].value_counts())

# Encode the sentiment labels
label_encoder = LabelEncoder()
movie_reviews_data['sentiment'] = label_encoder.fit_transform(movie_reviews_data['sentiment'])

# Text Preprocessing
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(movie_reviews_data)):
    print(i)
    review = BeautifulSoup(movie_reviews_data['review'][i], 'html.parser').get_text()  # Use BeautifulSoup to remove HTML tags
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

# Split the data into training (80%), testing (20%)
x_train, x_test, y_train, y_test = train_test_split(corpus, movie_reviews_data['sentiment'], test_size=0.2, random_state=40)

# Further split the training data into training (80%) and validation (20%) subsets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=40)

# Convert text data to numerical features using TF-IDF Vectorizer with n-grams
tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3))  # Adjust n-gram range as needed
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train).toarray()
x_val_tfidf = tfidf_vectorizer.transform(x_val).toarray()
x_test_tfidf = tfidf_vectorizer.transform(x_test).toarray()

# Create a neural network with dropout, batch normalization, and L2 regularization
model = Sequential()
model.add(Dense(units=512, activation='relu', input_dim=x_train_tfidf.shape[1], kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
custom_optimizer = Adam(learning_rate=0.001)  # Adjust learning rate
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Implement learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        return lr * 0.5
    return lr

lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model with early stopping and learning rate scheduler
history = model.fit(x_train_tfidf, y_train, epochs=30, batch_size=64, validation_data=(x_val_tfidf, y_val), callbacks=[early_stopping, lr_schedule])

# Plotting accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the test set
prediction = model.predict(x_test_tfidf)
pred = (prediction > 0.5).astype(int)

print("Accuracy (Test):", accuracy_score(y_test, pred))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, pred))
print("Classification Report (Test):\n", classification_report(y_test, pred))

# Save the model and vectorizer
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("movies_reviews_classification_model.pkl", "wb"))

# Load the model and vectorizer
loaded_tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
loaded_model = pickle.load(open('movies_reviews_classification_model.pkl', 'rb'))

# Function to test the model on new sentences
def test_model(sentence):
    sentence_vectorized = loaded_tfidf_vectorizer.transform([sentence]).toarray()
    prediction = loaded_model.predict(sentence_vectorized)
    res = (prediction > 0.5).astype(int)[0][0]
    if res == 1:
        return 'Positive review'
    else:
        return 'Negative review'

# Test the model with example sentences
sentence1 = 'This is a good movie'
result1 = test_model(sentence1)
print(result1)

sentence2 = 'This is the worst movie, I do not want to watch this again.'
result2 = test_model(sentence2)
print(result2)
