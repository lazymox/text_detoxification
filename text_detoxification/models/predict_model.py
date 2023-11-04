import joblib
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = '../models/logistic_regression_model.joblib'
TEST_DATA_PATH = '../data/interim/test.csv'

model = load(MODEL_PATH)

test_df = pd.read_csv(TEST_DATA_PATH)

X_test = test_df['reference']
y_test_tox = test_df['ref_tox']

y_pred_tox = model.predict(X_test)

accuracy = accuracy_score(y_test_tox, y_pred_tox)
f1 = f1_score(y_test_tox, y_pred_tox, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
