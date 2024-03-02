import os

import pandas as pd
import string

from keras.callbacks import EarlyStopping
from keras.optimizers.adamw import AdamW
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from scipy.stats import loguniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.optimizers import Adam

from src.my_util import Metrics


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


def create_model(input_dim, output_dim, lr=0.00005, decay=0.004):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=lr, weight_decay=decay), metrics=['accuracy'])
    return model


def main(config=None):
    # Load data
    train_df = pd.read_csv("data/raw/train.csv", delimiter=";")
    test_df = pd.read_csv("data/raw/test.csv", delimiter=";")

    # Clean text data
    train_df['cleaned_essay'] = train_df['essay'].apply(clean_text)
    test_df['cleaned_essay'] = test_df['essay'].apply(clean_text)

    int_encoder = LabelEncoder()
    int_labels = int_encoder.fit_transform(pd.concat([train_df['emotion'], test_df['emotion']]))
    train_df['int_emotion'] = int_labels[:len(train_df)]
    test_df['int_emotion'] = int_labels[len(train_df):]

    train_x = train_df['cleaned_essay']
    test_x = test_df['cleaned_essay']
    train_y = train_df['int_emotion']
    test_y = test_df['int_emotion']

    encoder = OneHotEncoder(sparse_output=False)
    one_hot_train_y = encoder.fit_transform(train_y.values.reshape(-1, 1))
    one_hot_test_y = encoder.fit_transform(test_y.values.reshape(-1, 1))

    vectorizer = TfidfVectorizer(use_idf=True, max_features=900)
    tf_idf_train_text = vectorizer.fit_transform(train_x).toarray()
    tf_idf_test_text = vectorizer.transform(test_x).toarray()
    input_dim = tf_idf_train_text.shape[1]
    output_dim = len(train_df['emotion'].unique())

    # Perform HP search
    if config is None:
        model = KerasClassifier(create_model, input_dim=input_dim, output_dim=output_dim)
        hp_space = {
            'lr': loguniform(0.00001, 0.001),
            'decay': loguniform(0.001, 0.2)
        }
        random_search = RandomizedSearchCV(estimator=model, param_distributions=hp_space, cv=5, n_iter=50)
        random_search_result = random_search.fit(tf_idf_train_text, one_hot_train_y, epochs=20, verbose=0)

        print(f"Best hyperparameters:")
        [print(f"\t{key} : {value}") for key, value in random_search_result.best_params_.items()]
        config = random_search_result.best_params_

    # Train the model with best parameters
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    best_model = create_model(input_dim=input_dim, output_dim=output_dim, **config)
    history = best_model.fit(
        tf_idf_train_text,
        one_hot_train_y,
        epochs=20,
        batch_size=32,
        validation_data=(tf_idf_test_text, one_hot_test_y),
        callbacks=[early_stopping]
    )

    # Plot training history
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("data/results/tf_idf/loss_curve")
    plt.close()

    # Evaluation
    os.makedirs("data/results/tf_idf", exist_ok=True)
    predicted_labels = best_model.predict(tf_idf_test_text)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    true_labels = np.argmax(one_hot_test_y, axis=1)
    metrics = Metrics(true_labels, predicted_labels)
    print(metrics.classification_report)
    cmd = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix)
    cmd.plot()
    plt.savefig("data/results/tf_idf/confusion_matrix")
    plt.close()


if __name__ == "__main__":
    main(
        # {"decay" : 0.028637005311523238, "lr" : 0.0003068396148794488}
    )