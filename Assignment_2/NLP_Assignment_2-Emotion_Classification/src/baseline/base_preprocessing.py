import pandas as pd
import string
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from gensim.models import Word2Vec
from keras import optimizers
import numpy as np


def load_data():
    df = pd.read_csv("F:/University/Courses/NLP/Assignments/Assignment_2/"
                     "NLP_Assignment_2-Emotion_Classification/data/raw/train.csv", delimiter=";",
                     names=['article_id', 'essay', 'emotion'])
    return df


def clean_text(doc):
    doc = doc.lower()
    for char in string.punctuation:
        doc = doc.replace(char, ' ')
    tokens = doc.split()
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    doc = " ".join(tokens)
    return doc


def compute_doc_vectors(docs, w2v_model):
    doc_vectors = []
    for doc in tqdm(docs):
        tokens = doc.split()
        doc_vector = np.zeros(w2v_model.vector_size)
        word_count = 0
        for token in tokens:
            if token in w2v_model.wv:
                doc_vector += w2v_model.wv[token]
                word_count += 1
        if word_count > 0:
            doc_vector /= word_count
        doc_vectors.append(doc_vector)
    return np.array(doc_vectors)


def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))

    model.add(Dense(1, activation='sigmoid'))
    return model


# Main function
def main():
    # Load data
    df = load_data()

    # Clean text data
    df['cleaned_essay'] = df['essay'].apply(clean_text)
    print(df.head(5))

    # Split the data into training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(df['cleaned_essay'], df['emotion'], test_size=0.3, random_state=42)

    # Train the word2vec model
    word2vec_model = Word2Vec(sentences=train_x, vector_size=100, window=5, min_count=1, workers=4)

    # Compute document vectors for training and validation sets
    train_x_vectors = compute_doc_vectors(train_x, word2vec_model)
    val_x_vectors = compute_doc_vectors(val_x, word2vec_model)

    # Build the classification model
    model = build_model(input_shape=(train_x_vectors.shape[1],))

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_x_vectors, train_y, epochs=10, batch_size=32, validation_data=(val_x_vectors, val_y))

    # Evaluate the model
    loss, accuracy = model.evaluate(val_x_vectors, val_y)
    print(f'Validation Accuracy: {accuracy*100:.2f}%')


if __name__ == "__main__":
    main()
