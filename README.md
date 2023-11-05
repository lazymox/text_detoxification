# Text Detoxification Project

Bexeiit Beibarys;
b.bexeiit@innopolis.university;
Exchange Student

## Overview
This project provides tools to detoxify text, aiming to remove or replace toxic language while preserving the original meaning as much as possible. It utilizes a BERT-based model fine-tuned for the detoxification task.

## Features
- Data preprocessing and cleaning.
- Training a logistic regression model for toxicity detection.
- Detoxification using a fine-tuned BART model.

## Installation
To set up the project environment:

```bash
git clone https://github.com/lazymox/text_detoxification
cd text_detoxification
pip install -r requirements.txt 
```
## Usage
Preprocessing Data
Run make_dataset.py to clean and preprocess your dataset:

```bash
python make_dataset.py
```

## Training the Model
To train the logistic regression model, use:

```bash
python train_model.py
```
The trained model will be saved as logistic_regression_model.joblib.

## Evaluating the Model
Evaluate the performance of the trained model with:

```bash
python predict_model.py
```

## Detoxifying Text
Use the following script to detoxify your text:

```bash
python main.py --text "Your text to detoxify here"
```

## NOTE: files 'train.csv' and 'filtered.tsv' were too large. 'filtered.tsv' should be in raw/data/...



