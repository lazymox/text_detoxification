import re
import string

import joblib

model = joblib.load('models/logistic_regression_model.joblib')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text


def text_to_features(text):
    return {text}


def detoxify_text(input_text):
    preprocessed_text = preprocess_text(input_text)

    features = text_to_features(preprocessed_text)
    prediction = model.predict(features)

    if prediction == 1:
        detoxified_text = detoxification(input_text)
    else:
        detoxified_text = input_text

    return detoxified_text


from transformers import BartForConditionalGeneration, AutoTokenizer

base_model_name = 'facebook/bart-base'
model_name = 'SkolkovoInstitute/bart-base-detox'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model_ = BartForConditionalGeneration.from_pretrained(model_name)


def detoxification(input_text):
    inputs = tokenizer([input_text], return_tensors="pt", max_length=1024, truncation=True)

    output_sequences = model_.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1024,
        num_return_sequences=1,
    )

    detoxified_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return detoxified_text


original_text = input()
detoxified = detoxify_text(original_text)
print(detoxified)
