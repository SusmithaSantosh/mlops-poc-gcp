import pandas as pd
import numpy as np
import pickle
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
# Load Titanic dataset from GCS
bucket_name = "mlops-poc-gcp"
file_name = "titanic.csv"
gcs_path = f"gs://{bucket_name}/{file_name}"
titanic_data = pd.read_csv(gcs_path)
# Data Preprocessing
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna('S', inplace=True)
titanic_data.drop(columns='Cabin', axis=1, inplace=True)
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
# Prepare Features and Labels
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']
# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)
# Save the trained model locally
model_filename = "model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)
# Evaluate Model
training_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))
print(f"Training Accuracy: {training_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
# Save metrics as JSON
metrics = {
    "training_accuracy": training_accuracy,
    "test_accuracy": test_accuracy
}
with open("metrics.json", "w") as file:
    json.dump(metrics, file)
# Upload Model and Metrics to Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
# Upload Model
blob = bucket.blob("models/model.pkl")
blob.upload_from_filename(model_filename)
# Upload Metrics
blob = bucket.blob("models/metrics.json")
blob.upload_from_filename("metrics.json")
print("Model and metrics uploaded to GCS successfully.")
