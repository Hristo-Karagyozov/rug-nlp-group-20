import pandas as pd
import string
from nltk.corpus import stopwords
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from keras.layers import Dense
# from keras.models import Sequential
# from keras import optimizers

df = pd.read_csv("F:/University/Courses/NLP/Assignments/Assignment_2/"
                 "NLP_Assignment_2-Emotion_Classification/data/train_ready_for_WS.csv")
print(df.head())
