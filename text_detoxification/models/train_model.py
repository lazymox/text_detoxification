import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from joblib import dump


def load_data(file_path):
    return pd.read_csv(file_path)


def train_model(X_train, y_train):
    pipeline = make_pipeline(
        TfidfVectorizer(sublinear_tf=True),
        LogisticRegression()
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))


def save_model(model, model_path):
    dump(model, model_path)


def main():
    train_file = '../data/interim/train.csv'
    val_file = '../data/interim/val.csv'

    train_df = load_data(train_file)
    val_df = load_data(val_file)

    X_train, y_train = train_df['translation'], train_df['trn_tox']
    X_val, y_val = val_df['translation'], val_df['trn_tox']

    model = train_model(X_train, y_train)

    evaluate_model(model, X_val, y_val)

    model_path = '../models/logistic_regression_model.joblib'
    save_model(model, model_path)
    print("Model training complete and saved to", model_path)




if __name__ == '__main__':
    main()
